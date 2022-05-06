# encoding=utf-8
from tkinter import Frame
from torch.utils.data import Dataset
#from utils.layout_data_processor import SortStrategy, LayoutDataProcessor
# from utils.data import load_layout
import torch
import os
import numpy as np
import pickle
from typing import List
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from script.rawdata_load import load_raw_data
from utils.path import remove_file,create_folder
from script.layout_process import LayoutProcessor, scale_with_format
from script.misc import ClassInfo,DataFormat,CategoryInfo
from model.img_encoding2 import get_model
'''
这个数据集（任务）和之前的不同之处：
图片需要编码
文字没有字数
'''

def get_raw_data(args,use_buffer = False):
    buffer_folder = os.path.join(args.buffer,'raw_data')
    create_folder(buffer_folder)
    buffer_filepath = os.path.join(buffer_folder, 'data.tmp')
    if use_buffer is False:
        remove_file(buffer_filepath)
    if os.path.exists(buffer_filepath):
        with open(buffer_filepath,'rb') as f:
            data = pickle.load(f)
        return data
    else:
        data = load_raw_data(args)
        with open(buffer_filepath,'wb') as f:
            pickle.dump(data, f)
        return data


def subsequent_mask(size):
    attn_shape = (1,size,size)
    subsequent_mask = np.triu(np.ones(attn_shape),k=1).astype('uint8') #  生成左下角为0（含对角），右上角为1的矩阵
    return torch.from_numpy(subsequent_mask) == 0 # 反转，左下角为True（含对角）的矩阵，右上False（表示遮盖）

category_info = CategoryInfo()

class Batch:
    def __init__(self, 
        batch_labels, 
        batch_bboxs, 
        batch_imgs, 
        batch_framework,
        pad,
        ):
        '''
        src_mask:(batch_size,1,seq_len)
        tgt_mask:(batch_size,seq_len,seq_len)
        src/tgt:(batch_size,seq_len)
        '''
        self.framework = batch_framework

        self.label = batch_labels
        self.bbox = batch_bboxs
        self.img = batch_imgs

        self.bbox_input = batch_bboxs[:,:-1,:]
        self.bbox_trg = batch_bboxs[:,1:,:]
        self.mask = self.make_std_mask(batch_labels, pad) # 'pad'+'sub_sequence' mask #注：使用（bbox_index，pad=0）会使得bos也会屏蔽掉，所以不使用这种
        self.n_tokens = (batch_labels != pad).data.sum()


        self.y = torch.tensor([category_info[frame['category']].id for frame in batch_framework])

    @staticmethod
    def make_std_mask(seq, pad):
        # create mask for decoder
        # represent 'pad'+'sub_sequence_mask'
        
        subseq_mask = (seq != pad).unsqueeze(-2)
        subseq_mask = subseq_mask & subsequent_mask(seq.size(-1)).type_as(subseq_mask.data)
        return subseq_mask

class LayoutDataset(Dataset):
    def __init__(
            self,
            args,
            data,
            mode='train'
    ):
        self.original_size = args.input_size
        self.layout_processor = LayoutProcessor(
            input_size = args.input_size,
            input_format = DataFormat.LTRB,
            grid_size = (args.grid_width,args.grid_height),
            grid_format = DataFormat.LTRB,
            num_classes = args.n_classes,
            )
        self.frameworks = data
        self.class_info = ClassInfo()
        self.size = 224
        self.other_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.imgs_f_folder = os.path.join(args.buffer,'imgs')
        # self.BOS = self.layout_processor.BOS
        # self.EOS = self.layout_processor.EOS #注：不需要EOS，添加EOS后续mask需要maskEOS和PAD两种
        self.PAD = self.layout_processor.PAD
        self.PAD_BOX = (0.0,0.0,0.0,0.0)
        self.PAD_IMG = torch.FloatTensor(1, 2048).fill_(0)
        self.n_modality = 3
        self.n_class = args.n_classes + 1 # 5+1 = 6

    def one_hot(self, label):
        size = list(label.size())
        size.append(self.n_class)
        label = label.reshape(-1)
        ones = torch.sparse.torch.eye(self.n_class)
        ones = ones.index_select(0,label)
        ones = ones.reshape(*size)
        return ones

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

    def __len__(self):
        return len(self.frameworks)

    def __getitem__(self, idx):
        return self.frameworks[idx]

    def collate_fn(self, batch):
        batch_framework = []
        batch_bboxs = []
        batch_labels = []
        batch_imgs = []
        batch_bbox_index = []
        batch_img_index = []
        max_n_tokens = 0 # n_tokens = n_element + 2


        for sample in batch:
            framework = sample
            '''
            {'category': 'travel', 
            'name': 'travel_0259', 
            'images_filepath': ['./dataset/MAGAZINE/images/travel\\travel_0259_1.png'], 
            'height': 300, 'width': 225, 'labels': ['text', 'image', 'headline'], 
            'polygens': [[...], [...], [...]], 
            'images_index': [0, 1, 0], 
            'keyword': ['guest'], 
            'bboxes': [(...), (...), (...)], 
            'bboxs_norm': [[...], [...], [...]]}
            '''
            img_filepath = os.path.join(self.imgs_f_folder, framework['name']+'.npy')
            num_element = len(framework['labels'])
            labels = []
            bboxs = []
            img_index = framework['images_index']
            bbox_index = []

            imgs = torch.from_numpy(np.load(img_filepath))
            for label, bbox in zip(framework['labels'], framework['bboxes']):
                v1,v2,v3,v4 = scale_with_format(
                    tuple(map(lambda x: float(x),bbox)),
                    old_size = self.original_size,
                    new_size = (1,1),
                    old_format = DataFormat.LTRB,
                    new_format = DataFormat.CWH
                )
                bboxs.append([v1,v2,v3,v4])
                labels.append(self.class_info[label].id)
                bbox_index.append(1)
            
            # [BOS] sentence [EOS] (actually use 'PAD')
            labels      = [self.PAD] + labels + [self.PAD]
            bbox_index  = [0] + bbox_index + [0]
            bboxs       = [self.PAD_BOX] + bboxs + [self.PAD_BOX]
            img_index   = [0] + img_index + [0]
            imgs        = torch.cat([self.PAD_IMG,imgs,self.PAD_IMG],dim=0)
            '''n_element -> n_tokens'''

            batch_framework.append(framework)       #[dic] 
            batch_labels.append(torch.tensor(labels))

            batch_bboxs.append(bboxs)               #[list]
            batch_bbox_index.append(torch.tensor(bbox_index))

            batch_imgs.append(imgs)                 #[tensor]
            batch_img_index.append(torch.tensor(img_index))
            if  max_n_tokens < num_element + 2:     # 一个批次的最大token数（包括bos，eos；不包括pad）
                max_n_tokens = num_element + 2
       
        # PAD
        batch_bbox_index = pad_sequence(batch_bbox_index, batch_first=True, padding_value=0)
        batch_img_index = pad_sequence(batch_img_index, batch_first=True, padding_value=0)
        batch_labels = pad_sequence(batch_labels, batch_first=True, padding_value=self.PAD).cpu().detach() # [bn,len]
        batch_bboxs = self._pad_box(batch_bboxs, max_n_tokens).cpu().detach()  # [bn,14,4]
        batch_imgs = self._pad_img(batch_imgs, max_n_tokens).cpu().detach() # [bn,len,2048]
        ''' the img in diff page:     [bn, n_tokens <= max_n_tokens, 2048] -> [bn, max_n_tokens, 2048]'''
        
        batch_labels = self.one_hot(batch_labels)
        return Batch(batch_labels, batch_bboxs, batch_imgs, batch_framework,pad=self.PAD)


