import os
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
from typing import List
from enum import Enum
from script.misc import ClassInfo, RenderMode
from utils.path import clear_folder, create_folder
from script.layout_process import masked_select



def compose_images(image_filepath:List,bboxes:List, background:Image):
    # image_filepath and bboxes is aligned with index
    # bboxes: e.g[(0,0,9,9),(10,10,50,50),...] x1y1x2y2 format
    # background_size: (width,height)
    # TODO:支持更多类型
    assert len(image_filepath) == len(bboxes)
    for file_path,boundding_box in zip(image_filepath,bboxes):
        x1 = boundding_box[0]
        y1 = boundding_box[1]
        x2 = boundding_box[2]
        y2 = boundding_box[3]
        width = x2-x1
        height = y2-y1
        if width == 0 or height ==0:
            assert '跳过处理'
        img = Image.open(file_path)
        img = img.resize((width,height))
        background.paste(img,(x1,y1))
    return background


class Render():
    def __init__(self, defalt_mode:RenderMode=RenderMode.SIMPLE):
        super(Render, self).__init__()
        self.class_info = ClassInfo()
        self.background_color = (200,200,200)
        self.defalt_mode = defalt_mode
    
    def __call__(self, framework):
        if self.defalt_mode == RenderMode.SIMPLE:
            return self.render_simple(framework)
        elif self.defalt_mode == RenderMode.IMAGE:
            return self.render_image(framework)
        elif self.defalt_mode ==RenderMode.IMAGEANDTEXT:
            return self.render_image_text(framework)
        else:
            return self.debug(framework)
    
    def compose_images(self, framework, background:Image, element_idx):
        for file_path,boundding_box in zip(framework['images_filepath'],masked_select(framework['bboxes'],element_idx)):
            # len(framework['images_filepath']) 可能不等于 len(masked_select(framework['bboxes'],framework['images.index']))
            x1 = boundding_box[0]
            y1 = boundding_box[1]
            x2 = boundding_box[2]
            y2 = boundding_box[3]
            width = x2-x1
            height = y2-y1
            if width <= 0 or height <=0:
                return background
            img = Image.open(file_path)
            img = img.resize((width,height))
            background.paste(img,(x1,y1))
        return background

    def compose_boxes(self, framework, background:Image, element_idx=None):
        image_draw = ImageDraw.Draw(background)
        if element_idx is None:
            for name,box in zip(framework['labels'],framework['bboxes']):
                color = self.class_info[name].color
                image_draw.rectangle(box,fill=None, outline=color, width=3)
                image_draw.text((box[0], box[1]), name, color)
        else:
            for name,box in zip(masked_select(framework['labels'], element_idx),masked_select(framework['bboxes'],element_idx)):
                color = self.class_info[name].color
                image_draw.rectangle(box,fill=None, outline=color, width=3)
                image_draw.text((box[0], box[1]), name, color)
        return background

    def compose_polygens(self, framework, background:Image, element_idx=None):
        image_draw = ImageDraw.Draw(background)
        if element_idx is None:
            for name,polygen in zip(framework['labels'], framework['polygens']):
                color = self.class_info[name].color
                image_draw.polygon(polygen,fill=None, outline=self.class_info[name].color, width=3)
                # image_draw.text(polygen[0], name, color)
        else:
            for name,box in zip(masked_select(framework['labels'], element_idx),masked_select(framework['polygens'],element_idx)):
                color = self.class_info[name].color
                image_draw.polygon(polygen,fill=None, outline=self.class_info[name].color, width=3)
                # image_draw.text(polygen[0], name, color)
        return background

    def render_simple(self, framework):
        # 绘制文本和图片的边框
        background_img = Image.new('RGBA', (framework['width'],framework['height']),self.background_color)
        page_image = self.compose_boxes(framework, background_img)
        return page_image

    def render_image(self, framework):
        # 绘制文本的边框 + 绘制图片内容
        background_img = Image.new('RGBA', (framework['width'],framework['height']),self.background_color)
        page_image = self.compose_images(framework, background_img, framework['images_index'])
        texts_index = list(map(lambda x: not bool(x), framework['images_index']))
        page_image = self.compose_boxes(framework, page_image, texts_index)
        return page_image
        
    def render_image_text(self, framework):
        # 将不同类别的文本表示 + 绘制图片内容
        assert 'not support'

    def debug(self,framework):
        # 绘制所有内容
        background_img = Image.new('RGBA', (framework['width'],framework['height']),self.background_color)
        page_image = self.compose_images(framework, background_img, framework['images_index'])
        page_image = self.compose_boxes(framework, page_image)
        page_image = self.compose_polygens(framework, page_image)
        return page_image


class LogPainter():
    def __init__(self, args, layout_processor, mode:RenderMode=RenderMode.SIMPLE, writer=None):
        super(LogPainter,self).__init__()
        self.save_path = os.path.join(args.log_root,'eval_log')
        create_folder(self.save_path)
        clear_folder(self.save_path)
        self.layout_processor = layout_processor
        self.writer = writer
        self.render = Render(mode)

    # 标签是类别也是图层先后顺序
    def plt_framework_comparison(self, framework1, framework2, title=''):
        # plt.figure(figsize=(10,5))
        fig = plt.suptitle(title)
        
        plt.subplot(1,2,1)
        plt.title('pred')
        plt.imshow(self.render(framework1))
        plt.axis('on')

        plt.subplot(1,2,2)
        plt.title('label')
        plt.imshow(self.render(framework2))
        plt.axis('on')
        return fig

    def log(self, framework, tensor_pred:List, name_prefix:str):
        title = name_prefix + framework['name']
        framework_pred = self.layout_processor.sent2framework(tensor_pred, framework)
        figure = self.plt_framework_comparison(framework_pred, framework, title)

        if self.writer is None:
            plt.savefig(os.path.join(self.save_path, f'{title}.png'))
        else:
            self.writer.add_figure(title, figure)
