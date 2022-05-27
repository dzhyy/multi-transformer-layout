from PIL import Image
import torch
import numpy as np
from torch import nn
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.models as models

from script.option import get_preprocess_args
from script.dataloader import get_raw_data
from utils.path import create_folder


class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()

    def forward(self, x):
        return torch.mean(x, dim=[-2, -1]) # [n,2048,7,10]->[n,2048]

def get_model():
    print('Loading 2D-ResNet-152 ...')
    model = models.resnet152(pretrained=True,)
    # test = list(model.children())[:-2]，去掉池化层和flatten层
    model = nn.Sequential(*list(model.children())[:-2], GlobalAvgPool())
    model.eval()
    print('loaded')
    return model


class FrameDataset(Dataset):
    def __init__(
            self,
            data,
    ):
        self.frameworks = data
        self.size = 224
        self.other_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def _img_transform(self, PILimg:Image):
        h = PILimg.height
        w = PILimg.width
        if isinstance(self.size, tuple) and len(self.size) == 2:
            height, width = self.size
        elif h >= w:
            height = int(h * self.size / w)
            width = self.size
        else:
            height = self.size
            width = int(w * self.size / h)
        
        img = PILimg.resize((width,height),Image.BILINEAR)
        return self.other_transform(img)

    def __len__(self):
        return len(self.frameworks)

    def __getitem__(self, idx):
        return self.frameworks[idx]


def run(args):
    device = torch.device('cpu') if args.cpu is True else torch.device('cuda:0')
    raw_data = get_raw_data(args,use_buffer=True)
    folder = os.path.join(args.buffer,'imgs')
    create_folder(folder)
    dataset = FrameDataset(raw_data)
    model = get_model().to(device)

    for framework in dataset:
        output_file = os.path.join(folder, framework['name']+'.npy')
        num_element = len(framework['labels'])
        imgs = torch.FloatTensor(num_element, 2048).fill_(0).to(device)

        img_index = framework['images_index']
        img_index2 = [idx for idx,item in enumerate(img_index) if item == 1]
        '''
        img_index: [1,0,1,0,0]
        img_index2: [0,2]
        '''
        for img_path,index in zip(framework['images_filepath'],img_index2): # imgs可能为空
            img = dataset._img_transform(Image.open(img_path)) # [3,224,360]
            img = img.unsqueeze(0).to(device)
            feature = model(img) # [1, 2048]
            imgs[index] = feature
        ''' the img in one page:     [n_pic <= n_element, 3,unknow,unkonw] -> [n_element, 2048]'''
        np.save(output_file, imgs.cpu().detach().numpy())

if __name__=='__main__':
    args = get_preprocess_args()
    run(args)