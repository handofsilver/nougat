
import os
import json, cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import defaultdict

from PIL import Image

class SortLayout:
    def __init__(self) -> None:
        self.kmeans2 = KMeans(n_clusters=2, n_init='auto')  # 指定聚类数目
        self.kmeans3 = KMeans(n_clusters=3, n_init='auto')  # 指定聚类数目

    def sort_by_column_and_y(self, img_box_tuples):
        '''根据 所在列.纵坐标 进行排序'''
        def get_box_num(img_box_tuple):
            box = img_box_tuple[1]
            box_num = int(box['column'])*10000000+int(box['y'])
            return box_num
        img_box_tuples = sorted(img_box_tuples, key=get_box_num)
        return img_box_tuples

    def get_relative_gap(self, img_box_tuples):
        '''计算每个目标的横坐标相对缩进'''
        tuples_by_column = defaultdict(list)
        for img, box in img_box_tuples:
            tuples_by_column[box['column']].append(box)
        for boxes in tuples_by_column.values():
            x_list = [box['x'] for box in boxes]
            min_x = min(x_list)
            for box, x in zip(boxes, x_list):
                # if box['label']=='公式': 
                #     box['x_gap'] = 0
                if box['label']=='参考文献': 
                    box['x_gap'] = 0
                else:
                    box['x_gap'] = x-min_x
                if box['x_gap']/box['width']>0.35 and box['label']!='公式': 
                    box['x_gap'] = 0

        # for img, box in img_box_tuples:
        #     if box['label']=='参考文献':
        #         for img, box in img_box_tuples:
        #             print(box['label'], box['column'], box['x_gap'], box['width'], img.size)

        return img_box_tuples

    def get_body_width(self, boxes):
        body_x1 = [box['x'] for box in boxes]
        body_x2 = [box['x'] + box['width'] for box in boxes]
        return max(body_x2) - min(body_x1)
    
    def get_body_height(self, boxes):
        body_y1 = [box['y'] for box in boxes]
        body_y2 = [box['y'] + box['height'] for box in boxes]
        return max(body_y2) - min(body_y1)

    def get_column_class(self, boxes, dominant_boxes, show=False):
        '''
        简介
            如果平均宽度超过页面宽度的50%则认为当前页面是单列的
        输入
            boxes: [
                {
                    'x': 280.6865485803,
                    'y': 311.7219834822,
                    'width': 1950.3642533454,
                    'height': 2921.09842832,
                    'label': '正文'
                    'txt': '正文内容'
                },
            ]
        return
            labels: 预测结果，0表示在第一列，1表示在第二列，2表示在第三列
            column_num: 1/2/3 表示该页面是几列排版
        '''
        if not boxes: return [], 0
        if not dominant_boxes: return [0]*len(boxes), 1
        heights = np.array([box['height'] for box in dominant_boxes])
        widths = np.array([box['width'] for box in dominant_boxes])
        body_width = self.get_body_width(dominant_boxes)         # 包含目标的宽度范围
        # 根据目标的平均宽度判断当前的排版是单列还是双列
        mean_width = sum((heights/sum(heights))*widths)     # 所有目标的加权平均宽度
        
        # 单列
        if mean_width/body_width>0.5:
            labels = np.array([0]*len(boxes))
            column_num = 1
        # 三列
        elif mean_width/body_width<0.33:
            # 获取聚类结果
            x_data = np.array([box['x']+box['width']/2 for box in boxes])
            x_data = x_data.reshape(-1, 1)
            self.kmeans3.fit(x_data)
            labels = self.kmeans3.labels_
            # 获取每个簇的中心点，若第一个簇在视觉上排在后面，则修改分类标签
            centers = self.kmeans3.cluster_centers_
            idx_to_sorted_idx = {
                idx: sorted_idx \
                for sorted_idx, (idx, _) in enumerate(
                    sorted(enumerate(centers), key=lambda x: x[1][0])
                )
            }
            labels = [idx_to_sorted_idx[label] for label in labels]
            column_num = 3
        # 双列
        else:
            # 获取聚类结果
            x_data = np.array([box['x']+box['width']/2 for box in boxes])
            x_data = x_data.reshape(-1, 1)
            self.kmeans2.fit(x_data)
            labels = self.kmeans2.labels_
            # 获取每个簇的中心点，若第一个簇在视觉上排在后面，则修改分类标签
            centers = self.kmeans2.cluster_centers_
            if centers[0][0]>centers[1][0]:
                labels = 1 - labels
            column_num = 2

        # 绘制数据点和簇中心
        if show:
            plt.scatter(x_data, np.zeros_like(x_data), c=labels)
            plt.scatter(centers, np.zeros_like(centers), marker='X', s=200, c='red')
            plt.show()

        return list(labels), column_num

    def sort_to_reading_order_by_cls(self, images, boxes):
        '''
        简介
            将目标识别结果按照类别分别排序
        输入
            images: 从原图中截取出来的 image 对象
                [image1, image2, ...]
            boxes: [
                {
                    'x': 280.6865485803,
                    'y': 311.7219834822,
                    'width': 1950.3642533454,
                    'height': 2921.09842832,
                    'label': '正文'
                    'txt': '正文内容'
                },
            ]
        return
            sorted_element_dic: {
                'head': [
                    (
                        image, 
                        {
                            'x': 280.6865485803,
                            'y': 311.7219834822,
                            'width': 1950.3642533454,
                            'height': 2921.09842832,
                            'label': '文章标题'
                            'txt': '文章标题内容'
                        }
                    ),
                ],
                'body': [
                    (
                        image,
                        {
                            'x': 280.6865485803,
                            'y': 311.7219834822,
                            'width': 1950.3642533454,
                            'height': 2921.09842832,
                            'label': '子标题'
                            'txt': '子标题内容'
                        }
                    ),
                ],
                ...
            }
        '''
        # 按照不同元素类型分组
        element_dic = {
            'title': [],
            'author': [],
            'catalog': [],
            'subtitle': [],
            'body': [],
            'footnote': [],
            'chart': [],
            'code': [],
            'fragment': [],
            'innernote': [],
            'sidenote': [],
            'footnote': [],
            'reference': [],
            'others': [],
        }
        if not boxes: return element_dic

        # 获得列编号
        main_boxes = [box for box in boxes if '注' not in box['label']]
        body_width = self.get_body_width(main_boxes)         # 包含目标的宽度范围
        body_height = self.get_body_height(main_boxes)       # 包含目标的高度范围
        
        # 拆分不同大类的目标，各自计算自己属于哪一列
        body_classes = ['文章标题', '作者', '副标题', '目录', '正文', '子标题', '公式', '间注', '注释', '参考文献']
        chart_classes = ['图片', '图片标题', '表格', '表格标题', '表格注释', '算法', '算法标题', '片段', '片段标题']
        body_images, body_boxes = [], []
        chart_images, chart_boxes = [], []
        other_images, other_boxes = [], []
        for img, box in zip(images, boxes):
            if box['label'] in body_classes:
                body_images.append(img)
                body_boxes.append(box)
            elif box['label'] in chart_classes:
                chart_images.append(img)
                chart_boxes.append(box)
            else:
                other_images.append(img)
                other_boxes.append(box)

        # print('post_detect_sort 216')
        # for box in body_boxes:
        #     print(box)
        # body_images = [img for img, box in zip(images, boxes) if box['label'] in body_classes] 
        # body_boxes = [box for box in boxes if box['label'] in body_classes]
        # 计算不同大类目标的列标签
        body_main_boxes = [box for box in body_boxes if box['label'] not in ['间注', '注释', '文章标题', '作者', '副标题']]
        body_column_labels, column_num = self.get_column_class(body_boxes, body_main_boxes)
        chart_column_labels, _ = self.get_column_class(chart_boxes, chart_boxes)
        other_column_labels, _ = self.get_column_class(other_boxes, other_boxes)

        images = body_images + chart_images + other_images
        boxes = body_boxes + chart_boxes + other_boxes
        column_labels = body_column_labels + chart_column_labels + other_column_labels
        
        # 根据列序号和坐标对检测结果进行排序
        for img, box, column in zip(images, boxes, column_labels):
            box['column'] = int(column)
            group_name = 'others'
            if box['label'] in ['文章标题']:
                group_name = 'title'
            elif box['label'] in ['作者']:
                group_name = 'author'
            elif box['label'] in ['目录']:
                group_name = 'catalog'
            # 放入子标题和公式以方便排序
            elif box['label'] in ['子标题', '正文', '公式', '注释']: 
                group_name = 'body'
            elif box['label'] in ['参考文献']:
                group_name = 'reference'
            # 在每个页面最后一个段落结束的地方插入图片标题和表格
            elif box['label'] in ['图片', '图片标题', '表格', '表格标题', '表格注释']:
                group_name = 'chart'
            elif box['label'] in ['算法', '算法标题', '代码', '代码标题']:
                group_name = 'code'
            # 片段分为可以直接当正文的和可以当注释的
            elif box['label'] in ['片段', '片段标题']:
                group_name = 'fragment'
            elif box['label'] in ['间注']:
                group_name = 'innernote'
            elif box['label'] in ['旁注']:
                group_name = 'sidenote'
            element_dic[group_name].append((img, box))
            # if box['label'] in ['注释']:
            #     element_dic['have_footnote'] = True

        # 在组内按照阅读顺序排序
        sorted_element_dic = {
            name: self.sort_by_column_and_y(group) for name, group in element_dic.items()
        }
        sorted_element_dic['body'] = self.get_relative_gap(sorted_element_dic['body'])
        sorted_element_dic['column_num'] = column_num
        sorted_element_dic['body_width'] = body_width
        sorted_element_dic['body_height'] = body_height
        
        # 提取出所有的子标题类别图片脚注类别图片
        sorted_element_dic['subtitle'] = [
            (img, box) for img, box in sorted_element_dic['body'] if box['label']=='子标题'
        ]
        sorted_element_dic['footnote'] = [
            (img, box) for img, box in sorted_element_dic['body'] if box['label']=='注释'
        ]
        return sorted_element_dic

    def sort_to_reading_order(self, images, boxes):
        '''
        简介
            将目标识别结果按照类别分别排序
        输入
            images: 从原图中截取出来的 image 对象
                [image1, image2, ...]
            boxes: [
                {
                    'x': 280.6865485803,
                    'y': 311.7219834822,
                    'width': 1950.3642533454,
                    'height': 2921.09842832,
                    'label': '正文'
                    'txt': '正文内容'
                },
                ...
            ]
        return
            sorted_element_list: [
                (
                    image, 
                    {
                        'x': 280.6865485803,
                        'y': 311.7219834822,
                        'width': 1950.3642533454,
                        'height': 2921.09842832,
                        'label': '文章标题',
                        'column': 0,
                        'txt': '文章标题内容'
                    }
                ),
                (
                    image,
                    {
                        'x': 280.6865485803,
                        'y': 611.7219834822,
                        'width': 1950.3642533454,
                        'height': 2921.09842832,
                        'label': '子标题'
                        'column': 0,
                        'txt': '子标题内容'
                    }
                ),
                ...
            ]
        '''
        # 按照不同元素类型分组
        element_dic = {
            'head': [],
            'body': [],
            'foot': [],
            'others': [],
        }
        if not boxes: return element_dic

        # 获得列编号
        main_boxes = [box for box in boxes if '注' not in box['label']]
        body_width = self.get_body_width(main_boxes)         # 包含目标的宽度范围
        body_height = self.get_body_height(main_boxes)       # 包含目标的高度范围
        
        # 拆分不同大类的目标，各自计算自己属于哪一列
        body_classes = ['文章标题', '作者', '副标题', '目录', '正文', '子标题', '公式', '间注', '注释', '参考文献']
        chart_classes = ['图片', '图片标题', '表格', '表格标题', '表格注释', '算法', '算法标题', '片段', '片段标题']
        body_images, body_boxes = [], []
        chart_images, chart_boxes = [], []
        other_images, other_boxes = [], []
        for img, box in zip(images, boxes):
            if box['label'] in body_classes:
                body_images.append(img)
                body_boxes.append(box)
            elif box['label'] in chart_classes:
                chart_images.append(img)
                chart_boxes.append(box)
            else:
                other_images.append(img)
                other_boxes.append(box)

        # 计算不同大类目标的列标签
        body_main_boxes = [box for box in body_boxes if box['label'] not in ['间注', '注释', '文章标题', '作者', '副标题']]
        body_column_labels, column_num = self.get_column_class(body_boxes, body_main_boxes)
        chart_column_labels, chart_column_num = self.get_column_class(chart_boxes, chart_boxes)
        other_column_labels, _ = self.get_column_class(other_boxes, other_boxes)

        if chart_column_num==1 and column_num==2 and chart_boxes[0]['x']>body_width/2:
            chart_column_labels = [1] * len(chart_column_labels)
            
        images = body_images + chart_images + other_images
        boxes = body_boxes + chart_boxes + other_boxes
        column_labels = body_column_labels + chart_column_labels + other_column_labels
        
        # 根据列序号和坐标对检测结果进行排序
        for img, box, column in zip(images, boxes, column_labels):
            box['column'] = int(column)
            group_name = 'others'
            if box['label'] in ['文章标题', '作者']:
                group_name = 'head'
            elif box['label'] in ['目录', '子标题', '正文', '公式', '参考文献']:
                group_name = 'body'
            elif box['label'] in ['图片', '图片标题', '表格', '表格标题', '表格注释']:
                group_name = 'body'
            elif box['label'] in ['算法', '算法标题', '代码', '代码标题']:
                group_name = 'body'
            elif box['label'] in ['片段', '片段标题']:
                group_name = 'body'
            elif box['label'] in ['间注']:
                group_name = 'body'
            # 放入子标题和公式以方便排序
            elif box['label'] in ['注释', '旁注']: 
                group_name = 'foot'
            # 片段分为可以直接当正文的和可以当注释的
            element_dic[group_name].append((img, box))

        # 在组内按照阅读顺序排序
        sorted_element_dic = {
            name: self.sort_by_column_and_y(group) for name, group in element_dic.items()
        }
        sorted_element_list = sorted_element_dic['head'] + sorted_element_dic['body'] + sorted_element_dic['foot']
        return sorted_element_list


def crop_image(img, bboxes):
    '''从原始图片中截取出每个识别框内的部分'''
    img_list = []
    # 循环处理每个标注框
    for box in bboxes:
        # print(box['label'])
        x = box["x"]
        y = box["y"]
        width = box["width"]
        height = box["height"]
        
        # 根据标注框的坐标和大小截取对应图片
        cropped_img = img.crop((x, y, x + width, y + height))
        
        img_list.append(cropped_img)
    return img_list

def read_json(file_path):
    '''读取json格式的一体存储的文件'''
    with open(file_path, 'r', encoding='utf8') as f:
        data_list = json.loads(f.read())
    return data_list

if __name__=='__main__':

    # idx2cls_path = 'D:\Work\算法模块/23-6-1 pdf 数据提取\code/idx2cls.json'
    # img_path = 'D:\Work\算法模块/23-6-1 pdf 数据提取\code/test_data/page_1.png'
    # label_path = 'D:\Work\算法模块/23-6-1 pdf 数据提取\code/test_data/page_1.json'
    
    # 读取类别名称映射、图片、图片标签
    idx2cls = read_json(idx2cls_path)
    image = Image.open(img_path)
    width, height = image.size  
    page_label = read_json(label_path)
    all_merged_texts_per_page = []  # 初始化存储每一页合并文本的列表
    # 将不同目标按照阅读顺序排序
    SL = SortLayout()
    for box in page_label['bbox']:
        box['label'] = idx2cls[box['label']]['name']
    sorted_element_dic = SL.sort_to_reading_order_by_cls(page_label['bbox'])
    for i, (name, boxes) in enumerate(sorted_element_dic.items()):
        if name == 'body':  # 只对body进行处理
            sub_texts = []  # 存储当前body的文本
            merged_texts_per_page = []
            for j, box in enumerate(boxes):
                if box['label'] == '文章标题' or box['label'] == '子标题' :#or box['label'] == '副标题':
                    sub_texts.append(f'"{box["label"]}": {box["txt"]}')
                elif box['label'] == '参考文献':
                    sub_texts.append(f'<{box["label"]}>\n{box["txt"]}\n</{box["label"]}>')  # 添加参考文献
                else:
                    sub_texts.append(box['txt'])  # 添加正文
                # 将当前 body 的文本合并
                merged_texts_per_page = '\n'.join(sub_texts)
                # 将合并后的文本存储到列表中
                all_merged_texts_per_page.append(merged_texts_per_page)
    # 按照阅读顺序将每个小目标存储下来
        # 按照阅读顺序将每个小目标存储下来
    for i, (name, boxes) in enumerate(sorted_element_dic.items()):
        sub_img_list = crop_image(image, boxes)
        #for j, sub_img in enumerate(sub_img_list):
        #    sub_img.save(os.path.join(out_path, f'{filename}_{i}_{name}_{j}.png'))

