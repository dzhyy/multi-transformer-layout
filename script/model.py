import os
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as transF
from typing import List
from PIL import Image
from modules.transformer import TransformerEncoder
from modules.img_encoding import get_model
from torch.nn.utils.rnn import pad_sequence
from script.layout_process import scale_with_format
from script.misc import ClassInfo,DataFormat


class Batch:
    def __init__(
        self, 
        label, 
        bbox, 
        img, 
        framework,
        num_class,
        pad,
        ):
        self.framework = framework
        self.orig_label = label[:, 1:-1]
        self.orig_bbox = bbox[:, 1:-1, :]

        self.label = self._make_one_hot(label[:, 1:], num_class)
        self.bbox = bbox[:,:-1,:]       # [bn,len,4]
        self.bbox_trg = bbox[:,1:,:]
        self.img = img[:,:-1,:]

        self.mask, self.pad_mask = self._make_std_mask(label[:, 1:], pad) # 'pad'+'sub_sequence' mask #注：使用（bbox_index，pad=0）会使得bos也会屏蔽掉，所以不使用这种
        self.n_tokens = (label != pad).data.sum()

    @staticmethod
    def _make_std_mask(seq, pad):

        def _subsequent_mask(size):
            attn_shape = (1,size,size)
            subsequent_mask = np.triu(np.ones(attn_shape),k=1).astype('uint8') #  生成左下角为0（含对角），右上角为1的矩阵
            return torch.from_numpy(subsequent_mask) == 0 # 反转，左下角为True（含对角）的矩阵，右上False（表示遮盖）
        
        # create mask for decoder
        # represent 'pad'+'sub_sequence_mask'
        pad_mask = (seq != pad).unsqueeze(-2)
        mask = pad_mask & _subsequent_mask(seq.size(-1)).type_as(pad_mask.data)
        return mask, pad_mask
    
    @staticmethod
    def _make_one_hot(label, n_class):
        size = list(label.size())
        size.append(n_class)
        label = label.reshape(-1)
        ones = torch.sparse.torch.eye(n_class)
        ones = ones.index_select(0,label)
        ones = ones.reshape(*size)
        return ones


class MULTModel(nn.Module):
    def __init__(self, hyp_params):
        """
        Construct a MulT model.
        l:box   (bn,14,4)
        a:img   (bn,14,2048)
        v:label (bn,14,6)
        """
        super(MULTModel, self).__init__()
        self.orig_d_l, self.orig_d_a, self.orig_d_v = 2048, 4, 6
        self.d_l, self.d_a, self.d_v = 30, 30, 30
        self.vonly = False
        self.aonly = True
        self.lonly = False
        self.num_heads = hyp_params.n_heads
                                    # default:
        self.layers = 5             # 5
        self.attn_dropout = 0.1     # 0.1
        self.attn_dropout_a = 0.0   # 0.0
        self.attn_dropout_v = 0.0   # 0.0
        self.relu_dropout = 0.1     # 0.1
        self.res_dropout = 0.1      # 0.1
        self.out_dropout = 0.0      # 0.0
        self.embed_dropout = 0.25   # 0.25
        self.bufferd_attn_mask = False      # True

        combined_dim = self.d_l + self.d_a + self.d_v

        self.partial_mode = self.lonly + self.aonly + self.vonly
        if self.partial_mode == 1:
            combined_dim = 3 * self.d_l   # assuming d_l == d_a == d_v
        else:
            combined_dim = 2 * (self.d_l + self.d_a + self.d_v)
        
        output_dim = 4

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)

        # 2. Crossmodal Attentions
        if self.lonly:
            self.trans_l_with_a = self.get_network(self_type='la')
            self.trans_l_with_v = self.get_network(self_type='lv')
        if self.aonly:
            self.trans_a_with_a = self.get_network(self_type='a')
            self.trans_a_with_l = self.get_network(self_type='al')
            self.trans_a_with_v = self.get_network(self_type='av')
        if self.vonly:
            self.trans_v_with_l = self.get_network(self_type='vl')
            self.trans_v_with_a = self.get_network(self_type='va')
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)
       
        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout   # 30,   0.1
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a # 30,   0.0
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v # 30,   0.0
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2*self.d_l, self.attn_dropout # 60,   0.1
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 3*self.d_a, self.attn_dropout #    0.1
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2*self.d_v, self.attn_dropout # 60,   0.1
        else:
            raise ValueError("Unknown network type")
        
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,   #0.1
                                  res_dropout=self.res_dropout,     # 0.1
                                  embed_dropout=self.embed_dropout, # 0.25
                                  attn_mask=self.bufferd_attn_mask)         # 0.1
            
    def forward(self, x_l, x_a, x_v, pad_mask, subseq_pad_mask):
        '''
        l:img   (bn,len,2048)
        a:box   (bn,len,4)
        v:label (bn,len,6)
        pad_mask: [bn,1,len]
        subseq_pad_mask: [bn,len,len]
        '''
        # [3,50,300]
        # [3,375,5]
        # [3,500,20]
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training) # [3,d1,len]
        x_a = x_a.transpose(1, 2)   # [3,d2,len]
        x_v = x_v.transpose(1, 2)   # [3,d3,len]
       
        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)   # [3,30,len]
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)   # [3,30,len]
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)   # [3,30,len]
        proj_x_a = proj_x_a.permute(2, 0, 1)    # [len,3,30]
        proj_x_v = proj_x_v.permute(2, 0, 1)    # [len,3,30]
        proj_x_l = proj_x_l.permute(2, 0, 1)    # [len,3,30]

        '''if self.lonly:
            # (V,A) --> L
            h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)
            h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)
            h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
            h_ls = self.trans_l_mem(h_ls)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = last_hs = h_ls[-1]   # Take the last output for prediction'''

        if self.aonly:
            # (L,V) --> A
            '''
            q: a
            kv: [a, l, v]

            '''
            h_a_with_as = self.trans_a_with_a(proj_x_a, proj_x_a, proj_x_a, subseq_pad_mask)    # mask:sub+pad    a passed    [18,2,30]
            h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l, pad_mask)           # mask:pad        l passed    [18,2,30]
            h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v, pad_mask)           # mask:pad        v passed    [18,2,30]
            h_as = torch.cat([h_a_with_as, h_a_with_ls, h_a_with_vs], dim=2)
            h_as = self.trans_a_mem(h_as)   # l&v atten self  [len, bn, 90]
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = last_hs = h_as.transpose(0, 1) # [bn, len, 90]

        '''if self.vonly:
            # (L,A) --> V
            h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l) # pass l
            h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a) # pass a
            h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
            h_vs = self.trans_v_mem(h_vs)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1]
        
        if self.partial_mode == 3:
            last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)'''
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        last_hs = self.out_layer(last_hs_proj)
        output = last_hs.sigmoid()
        return output, last_hs # ouput:[bn,len,4]


class LayoutMultiTransformer(nn.Module):
    def __init__(self, hyp_params):
        super().__init__()
        self.frame_size = hyp_params.frame_size
        self.img_resize = hyp_params.img_resize
        self.img_encoding = get_model()
        self.MultiTransformer = MULTModel(hyp_params)
        
        self.PAD = hyp_params.n_classes
        self.PAD_BOX = (0.0,0.0,0.0,0.0)
        self.PAD_IMG = torch.FloatTensor(1, 2048).fill_(0)
        self.n_classes = hyp_params.n_classes + 1 # [BOS&PAD] regard as same element
        self.class_info = ClassInfo()
    
    def _pad_img(self, imgs:List, max_len ,batch_first=True): #[n_samples,len,3,224,236] ->[n_samples,fix_len,3,224,360]:
        assert batch_first is True
        batch_imgs = []
        for img in imgs:
            for _ in range(img.size()[0], max_len):
                img = torch.cat([img, self.PAD_IMG],dim=0)
            batch_imgs.append(img)
        return torch.stack(batch_imgs)
                

    def _pad_box(self, bboxs:List, max_len, batch_first=True,):
        assert batch_first is True
        new_bboxs = []
        for box in bboxs:
            for _ in range(len(box),max_len):
                box = box + [self.PAD_BOX]
            new_bboxs.append(box)
        return torch.tensor(new_bboxs)

    def extract_feature(self, batch_samples:list, device):
        '''
        {'category': 'travel', 
        'name': 'travel_0259', 
        'images_filepath': ['./dataset/MAGAZINE/images/travel\\travel_0259_1.png'], 
        'height': 300, 'width': 225, 'labels': ['text', 'image', 'headline'], 
        'polygens': [[...], [...], [...]], 
        'images_index': [0, 1, 0], 
        'keyword': ['guest'], 
        'bboxes': [(...), (...), (...)], 
        '''
        batch_max_n_tokens = 0 # n_tokens = n_element + 2
        batch_bboxs = []
        batch_labels = []
        batch_imgs = []

        for sample in batch_samples:
            framework = sample
            # imgs process
            num_element = len(framework['labels'])
            imgs = torch.FloatTensor(num_element, 2048).fill_(0).to(device)
            img_index = framework['images_index']   # e.g: [1,0,1,0,0]
            img_index2 = [idx for idx,item in enumerate(img_index) if item == 1]    #e.g: [0,2] 
            for img_path,index in zip(framework['images_filepath'],img_index2): # imgs可能为空
                img = Image.open(img_path)
                h = img.height
                w = img.width
                if isinstance(self.img_resize, tuple) and len(self.img_resize) == 2:
                    height, width = self.img_resize
                elif h >= w:
                    height = int(h * self.img_resize / w)
                    width = self.img_resize
                else:
                    height = self.img_resize
                    width = int(w * self.img_resize / h)
                img = img.resize((width,height),Image.BILINEAR)
                img = transF.to_tensor(img)
                img = img.unsqueeze(0).to(device)
                feature = self.img_encoding(img) # [1, 2048]
                imgs[index] = feature   # the img in one page:     [n_pic <= n_element, 3,unknow,unkonw] -> [n_element, 2048]
            
            # label & bbox process
            labels = []
            bboxs = []
            bbox_index = []

            for label, bbox in zip(framework['labels'], framework['bboxes']):
                v1,v2,v3,v4 = scale_with_format(
                    tuple(map(lambda x: float(x),bbox)),
                    old_size = self.frame_size,
                    new_size = (1,1),
                    old_format = DataFormat.LTRB,
                    new_format = DataFormat.CWH
                )
                bboxs.append([v1,v2,v3,v4])
                labels.append(self.class_info[label].id)
                bbox_index.append(1)
            
            # [BOS] sentence [EOS] (regard as PAD)
            labels      = [self.PAD] + labels + [self.PAD]
            bbox_index  = [0] + bbox_index + [0]
            bboxs       = [self.PAD_BOX] + bboxs + [self.PAD_BOX]
            img_index   = [0] + img_index + [0]
            pad_img = self.PAD_IMG.to(device)
            imgs        = torch.cat([pad_img,imgs,pad_img],dim=0)# [n_element, 2048] -> [n_tokens, 2048]

            batch_labels.append(torch.tensor(labels))   # [tensor]
            batch_bboxs.append(bboxs)                   # [list]
            batch_imgs.append(imgs)                     # [tensor]
            if  batch_max_n_tokens < num_element + 2:   # 一个批次的最大token数（包括bos，eos；不包括pad）
                batch_max_n_tokens = num_element + 2
       
        # PAD
        batch_labels = pad_sequence(batch_labels, batch_first=True, padding_value=self.PAD).detach() # [bn,len]
        batch_bboxs = self._pad_box(batch_bboxs, batch_max_n_tokens).detach()  # [bn,14,4]
        batch_imgs = self._pad_img(batch_imgs, batch_max_n_tokens)                   # [bn,len,2048], keep gradient
        return Batch(
            label = batch_labels, 
            bbox = batch_bboxs, 
            img = batch_imgs, 
            framework = batch_samples, 
            num_class = self.n_class, 
            pad = self.PAD
            )

    def forward(self, batch, device):
        batch = self.extract_feature(batch, device)
        img, bbox, label = batch.img.to(device), batch.bbox.to(device), batch.label.to(device)
        pad_mask, mask = batch.pad_mask.to(device), batch.mask.to(device)
        output, _ = self.MultiTransformer(img, bbox, label, pad_mask, mask)
        return output, batch
