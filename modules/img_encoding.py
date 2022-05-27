
import torch as th
import torchvision.models as models
from torch import nn


class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()

    def forward(self, x):
        return th.mean(x, dim=[-2, -1]) # [n,2048,7,10]->[n,2048]

def get_model():
    print('Loading 2D-ResNet-152 ...')
    model = models.resnet152(pretrained=True,)
    # test = list(model.children())[:-2]，去掉池化层和flatten层
    model = nn.Sequential(*list(model.children())[:-2], GlobalAvgPool())
    model.eval()
    print('loaded')
    return model