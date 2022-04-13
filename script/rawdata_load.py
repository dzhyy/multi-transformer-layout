import logging
import os
import math
import xml.etree.ElementTree as ET
from typing import List
from script.misc import DataFormat


def load_raw_data(args) ->List[dict]:
    annotation_list = []
    break_count = 0
    for xmlfile in os.listdir(args.annotation_folder):
        annotation = {}
        # annotation: 
        # {
        # 'name':               e.g: fashion_0001
        # 'category': str       e.g: fashion
        # 'images_filepath':    e.g: ['./dataset/MAGAZINE/images/fashion\\fashion_0001_1.png', ...]
        # 'height': int         e.g: 300
        # 'width': int          e.g: 225
        # 'labels': List[str]   e.g: ['image', 'headline-over-image']
        # 'polygens':           e.g: [[(x0,y0),(x1,y1),(x2,y2)],[...]...] | use 'utils.layout_process.polytorec()' transform to boxes
        # 'keyword': Llist[str] e.g: ['seaon','bag']
        # 'images_index':        e.g: [1,0] | the image polygens index
        # }
        break_file = False
        tree = ET.parse(os.path.join(args.annotation_folder,xmlfile))
        root = tree.getroot()
        image_file_prefix = root.findall('filename')
        category = root.findall('category')
        size = root.findall('size')
        layout = root.findall('layout')
        text = root.findall('text')

        if len(image_file_prefix)!=1 or len(category)!=1 or len(size)!=1 or len(layout)!=1 or len(text) !=1:
            # break_file = True
            logging.warning(f'{xmlfile} skipped:find multiple element in xml.')
            break_count = break_count+1
            continue
        # category
        category = category[0].text
        annotation['category'] = category


        # image_file
        images = []
        image_file_prefix = image_file_prefix[0].text
        for idx in range(1,99):
            filename = image_file_prefix+f'_{idx}.png'
            image_filepath = os.path.join(args.img_folder,category)+'/'+filename
            if os.path.exists(image_filepath):
                images.append(image_filepath)
            else:
                break
        annotation['name'] = image_file_prefix
        annotation['images_filepath'] = images

        # size
        width = int(size[0].findall('width')[0].text)
        height = int(size[0].findall('height')[0].text)
        annotation['height'] = height
        annotation['width'] = width

        # polygens and label
        labels = []
        polygens = []
        img_index = []
        for element in layout[0].findall('element'):
            try:
                label = element.get('label')
                px = [int(i) for i in element.get('polygon_x').split(" ")]
                py = [int(i) for i in element.get('polygon_y').split(" ")]
                polygen = list(zip(px,py))
            except Exception as e:
                logging.warning(f'{xmlfile} skipped:{e}.')
                break_count = break_count+1
                break_file = True
                break
            if label == 'image':
                img_index.append(1)
            else:
                img_index.append(0)
            labels.append(label)
            polygens.append(polygen)
        if break_file:
            continue
        annotation['labels'] = labels
        annotation['polygens'] = polygens
        annotation['images_index'] = img_index
        
        # keywords
        keywords = []
        for element in text[0].findall('keyword'):
            keyword = element.text
            keywords.append(keyword)
        annotation['keyword'] = keywords
        
        '''
        judege and add boxes( x1y1x2y2 format)
        '''
        if sum(annotation['images_index']) !=len (annotation['images_filepath']):
            # break_file = True
            logging.warning(f'{xmlfile} skipped:The annotation is not aligned with the obtained picture info.skipped file.')
            break_count = break_count+1
            continue
        annotation['bboxes'] = getBoxes(annotation)

        annotation_list.append(annotation)
    logging.info(f'get {len(annotation_list)} pages, break page count={break_count}')
    return annotation_list


def getBoxes(annotation):
    # return list of layer, format:'x1y1x2y2'
    def check(annotation):  # 寻找text-over-image的下标
        the_index = []
        for i,label in enumerate(annotation['labels']):
            if label == 'text-over-image' or label =='headline-over-image':
                the_index.append(i)
        return the_index
    
    def poly2rec(points:list):
        # 取最小外接矩形
        # 'points':[(x0,y0),(x1,y1),(x2,y2),...,(xn,yn)]
        # return x1y1x2y2
        min_x = math.inf
        max_x = 0
        min_y = math.inf
        max_y = 0
        for i in range(len(points)):
            x = points[i][0]
            y = points[i][1]
            min_x = min(x,min_x)
            max_x = max(x,max_x)
            min_y = min(y,min_y)
            max_y = max(y,max_y)
        return min_x,min_y,max_x,max_y

    def polygen_extend(polygen):
        '''
        polygen:[(x1,y1),(x2,y2),(x3,y3)]
        '''
        new_polygen = []
        for idx in range(0,len(polygen)-1):
            new_polygen.append(polygen[idx])
            edge_points = []
            x1 = polygen[idx][0]
            y1 = polygen[idx][1]
            x2 = polygen[idx+1][0]
            y2 = polygen[idx+1][1]
            if x1-x2 == 0 : #竖线段
                if y2>=y1:
                    for y5 in range(y1//5+1,y2//5):
                        new_y = y5*5
                        new_x = x1
                        edge_points.append((new_x,new_y))
                else:
                    for y5 in range(y1//5-1,y2//5,-1):
                        new_y = y5*5
                        new_x = x1
                        edge_points.append((new_x,new_y))
            elif y1-y2 ==0: # 横线段
                if x2>=x1:
                    for x5 in range(x1//5+1,x2//5):
                        new_x = x5*5
                        new_y = y1
                        edge_points.append((new_x,new_y))
                else:
                    for x5 in range(x1//5-1,x2//5,-1):
                        new_x = x5*5
                        new_y = y1
                        edge_points.append((new_x,new_y))
            else: 
                continue #不添加新的点
            new_polygen.extend(edge_points)
        new_polygen.append(polygen[-1])
        return new_polygen
    
    # 没有异常（没有text-over-image元素，不会有异常）
    the_idx = check(annotation)
    polygens = annotation['polygens']
    if len(the_idx) == 0:
        return [poly2rec(polygen) for polygen in polygens]
    
    # 存在异常（原数据集对于over-xxx的元素，使用的是底层要素与text-over-image元素取并集，这与图片不符，实际打印出来问题明显）
    points = []
    for idx in the_idx:
        points.extend(polygen_extend(polygens[idx]))
    boxes = []
    for polygen in polygens: # 虽然只有图片内容需要处理，但是要维持元素的顺序
        polygen = polygen_extend(polygen)
        polygen_clip = list(set(polygen).difference(set(points))) # 若当前要素与text-over-image元素发生相交，先减去相交的点再取最小外接矩阵。
        if len(polygen_clip) != 0:
            box = poly2rec(polygen_clip)
            boxes.append(box)
        else:
            box = poly2rec(polygen) # 当前polygen为text-over-image元素本身
            boxes.append(box)
    return boxes