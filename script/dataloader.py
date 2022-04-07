# encoding=utf-8
from ast import keyword
from cv2 import transform
from torch.utils.data import Dataset
#from utils.layout_data_processor import SortStrategy, LayoutDataProcessor
# from utils.data import load_layout
import torch
import os
import numpy as np
import pickle
from PIL import Image
from typing import List
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from script.rawdata_load import load_raw_data
from utils.path import remove_file,create_folder
from script.layout_process import LayoutProcessor, scale_with_format
from script.misc import ClassInfo,DataFormat
'''
这个数据集（任务）和之前的不同之处：
图片需要编码
文字没有字数
'''

def subsequent_mask(size):
    attn_shape = (1,size,size)
    subsequent_mask = np.triu(np.ones(attn_shape),k=1).astype('uint8') #  生成左下角为0（含对角），右上角为1的矩阵
    return torch.from_numpy(subsequent_mask) == 0 # 反转，左下角为True（含对角）的矩阵，右上False（表示遮盖）


class Batch:
    def __init__(self, 
        batch_labels, 
        batch_bboxs, 
        batch_imgs, 
        batch_img_index, 
        batch_framework,
        pad,
        device
        ):
        '''
        src_mask:(batch_size,1,seq_len)
        tgt_mask:(batch_size,seq_len,seq_len)
        src/tgt:(batch_size,seq_len)
        '''
        self.framework = batch_framework
        self.label = batch_labels.to(device)

        '''模型的输出不能没有意义，x1x2y1y2的格式可能会出现(x1,y1)>(x2,y2)的情况，这样没有意义。
        因此bbox使用的格式是cxcywh的格式，无论何种输出都有意义'''
        self.bbox = batch_bboxs[:,:-1,:].to(device) # 即便在pad处也填充了值，但是会被mask掉
        self.bbox_trg = batch_bboxs[:,1:,:].to(device)

        self.img = batch_imgs # 不填充，按照索引使用
        self.img_index = batch_img_index.to(device)
        
        self.mask = self.make_std_mask(batch_labels, pad).to(device) # 'pad'+'sub_sequence' mask #注：使用（bbox_index，pad=0）会使得bos也会屏蔽掉，所以不使用这种
        self.seq_mask = (batch_labels != pad).to(device)
        self.n_tokens = self.seq_mask.data.sum()

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
            device,
            use_buffer = True,
            mode = 'train'
    ):
        self.device = device
        self.original_size = args.input_size
        self.layout_processor = LayoutProcessor(
            input_size = args.input_size,
            input_format = DataFormat.LTRB,
            grid_size = (args.grid_width,args.grid_height),
            grid_format = DataFormat.LTRB,
            num_classes = args.n_classes,
            )
        buffer_folder = os.path.join(args.log_root,'buffer')
        create_folder(buffer_folder)
        if use_buffer is False:
            remove_file(os.path.join(buffer_folder, 'data.tmp'))
        self.frameworks = self.preprocess(args, os.path.join(buffer_folder, 'data.tmp'))
        self.transform = transforms.Compose([
            transforms.Resize(size=(128, 128)),
            transforms.ToTensor(),
        ])
        self.class_info = ClassInfo()
        self.BOS = self.layout_processor.BOS
        # self.EOS = self.layout_processor.EOS #注：不需要EOS，添加EOS后续mask需要maskEOS和PAD两种
        self.PAD = self.layout_processor.PAD
        self.PAD_BOX = (0.5,0.5,1.0,1.0)
    
    def preprocess(self, args, buffer_filepath):
        if os.path.exists(buffer_filepath):
            with open(buffer_filepath,'rb') as f:
                data = pickle.load(f)
            return data
        else:
            data = load_raw_data(args)
            #for framework in data:
            #    framework['bboxs_norm']= self.layout_processor.framework2sent(framework)
            with open(buffer_filepath,'wb') as f:
                pickle.dump(data, f)
            return data

    def __len__(self):
        return len(self.frameworks)

    def __getitem__(self, idx):
        return self.frameworks[idx],

    def collate_fn(self, batch):
        batch_framework = []
        batch_bboxs = []
        batch_labels = []
        batch_imgs = []
        batch_bbox_index = []
        batch_img_index = []
        batch_imgs = []


        for item in batch:
            framework = item[0]
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
            bboxs = []
            labels = []
            imgs = []
            img_index = framework['images_index']
            bbox_index = []

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
            for img_path in framework['images_filepath']: # imgs可能为空
                img = self.transform(Image.open(img_path))
                imgs.append(img.to(self.device))

            # BOS EOS (actually PAD)
            bbox_index = [0] + bbox_index + [0]
            bboxs = [self.PAD_BOX] + bboxs + [self.PAD_BOX]
            img_index = img_index + [0]
            labels = labels + [self.PAD]

            batch_framework.append(framework)
            batch_labels.append(torch.tensor(labels))

            batch_bboxs.append(bboxs)
            batch_bbox_index.append(torch.tensor(bbox_index))

            batch_imgs.append(imgs)
            batch_img_index.append(torch.tensor(img_index))
       
        # PAD
        batch_labels = pad_sequence(batch_labels, batch_first=True, padding_value=self.PAD)
        batch_bbox_index = pad_sequence(batch_bbox_index, batch_first=True, padding_value=0)
        batch_img_index = pad_sequence(batch_img_index, batch_first=True, padding_value=0)
        batch_bboxs = pad_box(batch_bboxs, batch_first= True,padding_box=self.PAD_BOX)
        
        return Batch(batch_labels, batch_bboxs ,batch_imgs, batch_img_index, batch_framework,pad=self.PAD,device=self.device)


def pad_box(sentences:List,batch_first,padding_box=(0.0,0.0,1.0,1.0)):
    assert batch_first is True
    max_len = 0
    for sentence in sentences:
        if len(sentence)>max_len:
            max_len = len(sentence)
    new_sentences = []
    for sentence in sentences:
        for _ in range(len(sentence),max_len):
            sentence = sentence + [padding_box]
        new_sentences.append(sentence)
    return torch.tensor(new_sentences)
        