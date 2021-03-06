from enum import Enum
import torch.distributed as dist

class DataFormat(Enum):
    CWH = 1     # 中心宽高
    LTRB = 2    # x1y1x2y2
    LTWH = 3    # 左上角宽高

class RenderMode(Enum):
    BOX = 1
    IMAGE = 2
    IMAGEANDTEXT = 3
    DEBUG = 4

class Element():
    def __init__(self,name,id,color='#FF000088') -> None:
        self.name = name
        self.id = id
        self.color = color

class CategoryInfo(object):
    # "fashion"'food',"news"，"science",'travel'，'wedding'
    def __init__(self) -> None:
        self.fashion = Element('fashion',0,'#FF000088')
        self.food = Element('food',1,'#00FF0088')
        self.news = Element("news",2,'#0000FF88')
        self.science = Element("science",3,'#FFFF0088')
        self.travel = Element('travel',4,'#00FFFF88')
        self.wedding = Element('wedding',5,'#0088FFFF')
        self.n_classes = 6
        self.name_map={
            self.fashion.name: self.fashion,
            self.food.name: self.food,
            self.news.name: self.news,
            self.science.name: self.science,
            self.travel.name: self.travel,
            self.wedding.name: self.wedding

        }
        self.id_map={
            self.fashion.id: self.fashion,
            self.food.id: self.food,
            self.news.id: self.news,
            self.science.id: self.science,
            self.travel.id: self.travel,
            self.wedding.id: self.wedding
        }
    
    def __getitem__(self,id):
        if isinstance(id, str):
            return self.name_map[id]
        elif isinstance(id, int):
            if id >self.n_classes:
                return self.id_map[self.unknown_class.id]
            return self.id_map[id]


class ClassInfo(object):
    # "text"，'image',"headline"，"text-over-image",'headline-over-image'，5
    def __init__(self) -> None:
        self.text = Element('text',0,'#FF000088')
        self.image = Element('image',1,'#00FF0088')
        self.headline = Element("headline",2,'#0000FF88')
        self.text_over_image = Element("text-over-image",3,'#FFFF0088')
        self.headline_over_image = Element('headline-over-image',4,'#00FFFF88')
        self.unknown_class = Element('unknown',5,'#0088FFFF')
        self.n_classes = 5
        self.name_map={
            self.text.name: self.text,
            self.image.name: self.image,
            self.headline.name: self.headline,
            self.text_over_image.name: self.text_over_image,
            self.headline_over_image.name: self.headline_over_image,
            self.unknown_class.name: self.unknown_class

        }
        self.id_map={
            self.text.id: self.text,
            self.image.id: self.image,
            self.headline.id: self.headline,
            self.text_over_image.id: self.text_over_image,
            self.headline_over_image.id: self.headline_over_image,
            self.unknown_class.id: self.unknown_class
        }
    
    def __getitem__(self,id):
        if isinstance(id, str):
            return self.name_map[id]
        elif isinstance(id, int):
            if id >self.n_classes:
                return self.id_map[self.unknown_class.id]
            return self.id_map[id]


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0