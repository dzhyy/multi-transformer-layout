# encoding=utf-8
from torch.utils.data import Dataset
import torch
import os
import pickle
from typing import List
from torch.utils.data import Dataset
from script.rawdata_load import load_raw_data
from utils.path import remove_file,create_folder
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


class LayoutDataset(Dataset):
    def __init__(
            self,
            args,
            use_buffer=False,
            mode='train'
    ):
        self.original_size = args.frame_size
        self.frameworks = get_raw_data(args, use_buffer)
        self.n_modality = 3

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
        return batch    # for debug
