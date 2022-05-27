import os
import itertools
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
from script.misc import ClassInfo, RenderMode
from utils.path import clear_folder, create_folder
from script.layout_process import masked_select
import numpy as np


class Render():
    def __init__(self, defalt_mode:RenderMode=RenderMode.BOX):
        super(Render, self).__init__()
        self.class_info = ClassInfo()
        self.background_color = (200,200,200)
        self.defalt_mode = defalt_mode
    
    def __call__(self, framework):
        if self.defalt_mode == RenderMode.BOX:
            return self.render_simple(framework)
        elif self.defalt_mode == RenderMode.IMAGE:
            return self.render_image(framework)
        elif self.defalt_mode ==RenderMode.IMAGEANDTEXT:
            return self.render_image_text(framework)
        elif self.defalt_mode ==RenderMode.IMAGEANDTEXT:
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


class Painter():
    def __init__(self, save_path, mode:RenderMode=RenderMode.BOX, writer=None):
        super(Painter,self).__init__()
        self.save_path = save_path
        self.writer = writer
        self.render = Render(mode)

    def clean_savepath(self):
        create_folder(self.save_path)
        clear_folder(self.save_path)

    def _plt_framework(self, framework, title=''):
        # plt.figure(figsize=(10,5))
        fig = plt.suptitle(title)
        
        plt.title('pred')
        img = self.render(framework)
        plt.imshow(img)
        plt.axis('on')
        return fig 

    # 标签是类别也是图层先后顺序
    def _plt_framework_comparison(self, framework1, framework2, title=''):
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

    def draw(self, framework1, framework2=None, name_prefix:str=''):
        title = name_prefix + framework1['name']
        if framework2 is None:
            figure = self._plt_framework(framework1, title)
        else:
            figure = self._plt_framework_comparison(framework1, framework2, title)

        if self.writer is None:
            plt.savefig(os.path.join(self.save_path, f'{title}.png'))
        else:
            self.writer.add_figure(title, figure)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
 
    print(cm)
 
    fig = plt.figure()
 
 
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
 
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
 
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig