from script.misc import DataFormat
import numpy as np
from typing import Tuple, List
from script.misc import ClassInfo
import torch
'''
以MAGAZINE数据集为例，数据经历以下四个阶段
( frame: poly->rec ) -> for [layer] -> grid
'''

def calculate_vocab_size(
    num_classes: int,
):
    return num_classes + 2     # 这里的3是特殊字符。


def calculate_bos_token_index(vocab_size: int):
    return vocab_size - 2 


def calculate_pad_token_index(vocab_size: int):
    return vocab_size - 1



def to_cwh_format(
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float
    ):
    width = max_x - min_x
    height = max_y - min_y
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    return center_x, center_y, width, height


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def scale(bbox,old_size:tuple,new_size:tuple):
    v1,v2,v3,v4 = bbox
    ox, oy = old_size
    nx, ny = new_size
    v1 = v1 / ox
    v2 = v2 / oy
    v3 = v3 / ox
    v4 = v4 / oy

    # 0-1的值放缩回new  w和h
    if (nx > 1):
        v1 = round(v1 * nx)
        v3 = round(v3 * nx)
    else:
        v1 = round(v1,4)
        v3 = round(v3,4)
    if (ny > 1):
        v2 = round(v2 * ny)
        v4 = round(v4 * ny)
    else:
        v2 = round(v2,4)
        v4 = round(v4,4)

    return v1, v2, v3, v4




def scale_with_format(
    element_bbox:tuple,
    old_size:tuple,
    new_size:tuple = (1,1), 
    old_format:DataFormat = DataFormat.LTRB, 
    new_format:DataFormat = DataFormat.LTRB
    ):
    '''
    input format: only support x1y1x2y2 format
    output format: 
    '''
    # TODO: 转换grid支持（LTRB->LTRB,LTWH，CWH），从grid转回（仅支持LTRB->LTRB），低优先级
    assert len(element_bbox) == 4
    assert len(old_size) == 2
    assert len(new_size) == 2
    assert old_format == DataFormat.LTRB  # TODO 支持更多类型
    xmin,ymin,xmax,ymax = map(lambda x: float(x),element_bbox)
    frame_width, frame_height = old_size
    scale_width, scaled_height = new_size

    # 缩放约0-1的值，且支持越界
    if (new_format == DataFormat.CWH):
        center_x, center_y, width, height = to_cwh_format(xmin, ymin,xmax, ymax)
        v1 = center_x / frame_width
        v2 = center_y / frame_height
        v3 = width / frame_width
        v4 = height / frame_height
    elif (new_format == DataFormat.LTWH):
        center_x, center_y, width, height = to_cwh_format(xmin, ymin, xmax, ymax)
        v1 = xmin / frame_width
        v2 = ymin / frame_height
        v3 = width / frame_width
        v4 = height / frame_height        # 这里都是占比，均小于1
    elif (new_format == DataFormat.LTRB):
        v1 = xmin / frame_width
        v2 = ymin / frame_height
        v3 = xmax / frame_width
        v4 = ymax / frame_height
    else:
        raise AssertionError('Not supported data format!')
    
    # 0-1的值放缩回new  w和h
    if (scale_width > 1):
        v1 = round(v1 * scale_width)
        v3 = round(v3 * scale_width)
    else:
        v1 = round(v1,4)
        v3 = round(v3,4)
    if (scaled_height > 1):
        v2 = round(v2 * scaled_height)
        v4 = round(v4 * scaled_height)
    else:
        v2 = round(v2,4)
        v4 = round(v4,4)

    return v1, v2, v3, v4


def masked_select(list:List,mask:List):
    # mask: [1,0,0]
    # list: [e1, e2, e3]
    # return->[(0, 0, 225, 300)]
    assert len(list) == len(mask)
    mask = map(bool, mask)
    new_list = []
    for e,flag in zip(list,mask):
        if flag:
            new_list.append(e)
    return new_list
    

class LayoutProcessor():
    def __init__(
        self,
        input_size: Tuple[int],
        input_format: DataFormat,
        grid_size: Tuple[int] = (32,32),
        grid_format: DataFormat = DataFormat.LTRB,
        num_classes: int = 5,
    ):
        '''
        only support x1y1x2y2 format layout data as input
        contation grid generation, grid projection
        for (v1,v2,v3,v4): bbox<----------bbox2grid()-------->grid<------grid_project()------>projected grid
        for framework<>sentence:   
        '''
        assert input_format == DataFormat.LTRB
        super(LayoutProcessor, self).__init__()
        self.grid_width,self.grid_height = grid_size
        self.input_width,self.input_height = input_size  # TODO:support diff input size 
        self.num_classes = num_classes
        self.input_format = input_format
        self.grid_format = grid_format
        self.vocab_size = calculate_vocab_size(num_classes)
        self.class_info = ClassInfo()
        self.BOS = calculate_bos_token_index(self.vocab_size)
        # self.EOS = calculate_eos_token_index(self.vocab_size)
        self.PAD = calculate_pad_token_index(self.vocab_size)

    def framework2sent(self, framework):
        '''
        {'category': 'fashion', 
        'name': 'fashion_0001', 
        'images_filepath': ['./dataset/MAGAZINE/images/fashion\\fashion_0001_1.png'], 
        'height': 300, 
        'width': 225, 
        'labels': ['image', 'headline-over-image'], 
        'polygens': [[...], [...]], 
        'images_index': [1, 0], 
        'keyword': [], 
        'bboxes': [(...), (...)]}
        '''
        bboxs = []
        for bbox in framework['bboxes']:
                v1,v2,v3,v4 = scale_with_format(
                tuple(map(lambda x: float(x),bbox)),
                old_size = (self.input_width,self.input_height),
                new_size = (1,1),
                old_format = self.input_format,
                new_format = DataFormat.LTRB
                )
                bboxs.append([v1,v2,v3,v4])
        return bboxs
            

    def sent2framework(self, sent:List[int], base_framework):
        # sent: [0,p1,p2,p3,p4,2...]
        n_elements = len(base_framework['labels'])
        sent = sent[:n_elements*4] # 预测值最后几位可能不是PAD，且PAD可能出现在中间几位，这里根据n_elements做裁剪可去掉本应为PAD和EOS的we
        framework = {}
        grids = []
        bboxes = []
        pos = 0
        for i in range(n_elements):
            # id = sent[pos]
            v1 = sent[pos]
            v2 = sent[pos+1]
            v3 = sent[pos+2]
            v4 = sent[pos+3]
            pos = pos + 4

            # labels.append(self.class_info[id].name)
            # image_index.append(int(id == self.class_info.image.id))
            v1,v2,v3,v4 = self.grid_project((v1,v2,v3,v4),backward=True)
            grids.append((v1,v2,v3,v4))
            v1,v2,v3,v4 = self.bbox2grid((v1,v2,v3,v4),backward=True)
            bboxes.append((v1,v2,v3,v4))
        
        # framework['labels'] = labels
        # framework['images_index'] = image_index
        framework['bboxes'] = bboxes
        framework['grid'] = grids
        framework.update({k:v for k,v in base_framework.items() if k not in framework})

        return framework
        
    '''
    bbox
    '''
    def bbox2grid(
        self,
        bbox: tuple,
        backward:bool = False
    ):
        '''
        gridding from boudding box:
        '''
        if not backward:
            v1,v2,v3,v4 = scale_with_format(
                tuple(map(lambda x: float(x),bbox)),
                old_size = (self.input_width,self.input_height),
                new_size = (self.grid_width,self.grid_height),
                old_format = self.input_format,
                new_format = self.grid_format
                )
        else:
            v1,v2,v3,v4 = scale_with_format(
                tuple(map(lambda x: float(x),bbox)),
                old_size = (self.grid_width,self.grid_height),
                new_size = (self.input_width,self.input_height),
                old_format = self.grid_format,
                new_format = self.input_format
                )
        
        return v1, v2, v3, v4


    def grid_project(self, 
        input: tuple, 
        backward:bool = False
    ):
        '''
        grid_vec: as an instance of DataFormat
        grid_width&grid_height: the max grid border
        bias: num_classes
        '''
        v1,v2,v3,v4 = input
        if not backward:
            # v1,v3∈[0,grid_width]
            # v2,v4∈[0,grid_height]
            v1, v3 = np.clip([v1, v3], 0, self.grid_width)
            v2, v4 = np.clip([v2, v4], 0, self.grid_height)   # TODO:添加对越界元素的支持

            v1 = v1 + self.num_classes
            v2 = v2 + self.grid_width + self.num_classes
            v3 = v3 + self.grid_width + self.grid_height + self.num_classes
            v4 = v4 + self.grid_width + self.grid_height + self.grid_width + self.num_classes
            return v1, v2, v3, v4
        else:
            v1 = v1 - self.num_classes
            v2 = v2 - self.grid_width - self.num_classes
            v3 = v3 - self.grid_width - self.grid_height - self.num_classes
            v4 = v4 - self.grid_width - self.grid_height - self.grid_width - self.num_classes
            return v1, v2, v3, v4


