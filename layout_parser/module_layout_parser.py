import os, torch
import numpy as np

from time import time
from tqdm import tqdm
from PIL import Image
from collections import defaultdict
from ditod import add_vit_config
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

from post_detect_sort_target import SortLayout
from post_detect_reshape import remove_broken_lines

def calculate_iou(box_a, box_b):
    '''
    计算两个方框的交并比
    '''
    # 解包方框坐标和尺寸
    x1, y1, width1, height1 = box_a
    x2, y2, width2, height2 = box_b
    
    # 计算交的坐标
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + width1, x2 + width2)
    inter_y2 = min(y1 + height1, y2 + height2)
    
    # 计算交的宽度和高度
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    
    # 如果交的宽度和高度都大于0，则计算交的面积
    if inter_width > 0 and inter_height > 0:
        intersection_area = inter_width * inter_height
    else:
        intersection_area = 0
    
    # 计算并的面积
    area_a = width1 * height1
    area_b = width2 * height2
    union_area = area_a + area_b - intersection_area
    
    # 计算IoU
    iou = intersection_area / union_area if union_area != 0 else 0
    iou_a = intersection_area / area_a
    iou_b = intersection_area / area_b
    
    return iou, iou_a, iou_b

class BatchedLayoutPredictor(DefaultPredictor):
    def __init__(self, cfg):
        super().__init__(cfg)

    def __call__(self, batch_images):
        # print('self.input_format', self.input_format)
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                batch_images = [img[:, :, ::-1] for img in batch_images]
            shape_list = [img.shape[:2] for img in batch_images]
            batch_images = [self.aug.get_transform(img).apply_image(img) for img in batch_images]
            batch_images = [torch.as_tensor(img.astype("float32").transpose(2, 0, 1)) for img in batch_images]
            [img.to(self.cfg.MODEL.DEVICE) for img in batch_images]
            inputs = [
                {"image": img, "height": shape[0], "width": shape[1]} \
                    for img, shape in zip(batch_images, shape_list)
            ]

            predictions = self.model(inputs)
            predictions = [tmp["instances"]._fields for tmp in predictions]
            # print(predictions[0])
            for prediction in predictions:
                prediction['pred_boxes'] = prediction['pred_boxes'].tensor.cpu()
                prediction['scores'] = prediction['scores'].cpu()
                prediction['pred_classes'] = prediction['pred_classes'].cpu()
                del prediction['pred_masks']
            # print(predictions[0])
        torch.cuda.empty_cache()
        return predictions

class LayoutParser:
    def __init__(
        self,
        config_file, 
        model_path, 
        idx2cls, 
        device
    ):
        cfg = get_cfg()  
        add_vit_config(cfg)
        cfg.merge_from_file(config_file)
        cfg.merge_from_list(['MODEL.WEIGHTS', model_path])
        cfg.MODEL.DEVICE = device
        self.layout_predictor = BatchedLayoutPredictor(cfg)
        self.layout_predictor.model.eval()
        self.idx2cls = idx2cls # 类别编号到名称的映射
        self.sort_layout = SortLayout()
    
    def save_target_img(self, path_dict_list, batch_target_img_box_dicts, debug=False):
        '''
        输入
            path_dict_list: [
                {
                    'pdf_file': '/the/path/of.pdf',
                    'out_path': '/path/of/result',
                    'page_num': int
                },
                ...
            ]
            batch_target_img_box_dicts: [
                [
                    {
                        'image': target_img,
                        'bbox': {
                            'x': x,
                            'y': y,
                            'width': width,
                            'height': height,
                            'label': class_name
                        }
                    },
                    ...
                ],
                ...
            ]
        '''
        if not debug: return
        for path_dict, target_img_box_dicts in zip(path_dict_list, batch_target_img_box_dicts):
            for i, target_dict in enumerate(target_img_box_dicts):
                img_path = os.path.join(
                    path_dict['out_path'], 
                    f"{path_dict['page_num']}"
                )
                if not os.path.exists(img_path):
                    os.mkdir(img_path)
                img_path = os.path.join(
                    img_path, 
                    f"{target_dict['bbox']['label']}_{i}.png"
                )
                target_dict['image'].save(img_path)

    def deduplicate(self, target_img_box_dicts):
        '''
        简介
            如果有两个目标框高度重叠，则保留置信度最高的目标框
        输入
            target_img_box_dicts: [
                {
                    'image': target_img,
                    'bbox': {
                        'x': x,
                        'y': y,
                        'width': width,
                        'height': height,
                        'label': class_num,
                        'score': float
                    }
                },
                ...
            ]
        输出
        '''
        new_target_img_box_list = []
        boxes = [
            (tmp['bbox']['x'], tmp['bbox']['y'], tmp['bbox']['width'], tmp['bbox']['height']) \
            for tmp in target_img_box_dicts
        ]
        for i, (box1, img_box) in enumerate(zip(boxes, target_img_box_dicts)):
            for box2, tmp_img_box in zip(boxes[i+1:], target_img_box_dicts[i+1:]):
                iou, iou_1, iou_2 = calculate_iou(box1, box2)
                # 保留概率更大的
                if iou>0.7:
                    if img_box['bbox']['score']<tmp_img_box['bbox']['score']:
                        img_box['bbox']['delete']=True
                        break
                    else:
                        tmp_img_box['bbox']['delete']=True
                # 保留面积更大的
                elif iou_1>0.9:
                    img_box['bbox']['delete']=True
                    break
                elif iou_2>0.9:
                    tmp_img_box['bbox']['delete']=True
        new_target_img_box_list = [tmp for tmp in target_img_box_dicts if not tmp['bbox'].get('delete', False)]
        return new_target_img_box_list

    def parse_batch_pdf_image(self, batch_images, detect_results):
        '''
        简介
            解析模型识别的结果
        输入
            batch_images: [Image, Image, ...]
            detect_results: 
        输出
            batch_target_img_box_dicts: [
                [
                    {
                        'image': target_img,
                        'bbox': {
                            'x': x,
                            'y': y,
                            'width': width,
                            'height': height,
                            'label': class_num,
                            'score': float
                        }
                    },
                    ...
                ],
                ...
            ]
        '''
        batch_target_img_box_dicts = []
        # res_list = [detect_result["instances"]._fields for detect_result in detect_results]
        # valid_detect_list = [res['scores'] > 0.5 for res in res_list]
        # for res, valid_detect in zip(res_list, valid_detect_list):
        #     [res[key] = res[key][valid_detect,...] for key in res]
        for img, result in zip(batch_images, detect_results):
            # print('140', output)
            # result = detect_result["instances"]._fields
            # print('142', res)
            # 筛选出置信度大于0.5的候选框
            valid_detect = result['scores'] > 0.5
            for key in result:
                result[key] = result[key][valid_detect,...]
            # 提取每一张图片中的目标相关信息
            target_img_box_dicts = [] 
            pred_boxes = result['pred_boxes']
            pred_classes = result['pred_classes']
            pred_scores = result['scores']
            offset = min(img.shape[0], img.shape[1]) * 0.005
            for i in range(len(pred_classes)):
                class_name = self.idx2cls[str(pred_classes[i].item()+1)]
                w_min = round(pred_boxes[i][0].item() - offset)
                w_max = round(pred_boxes[i][2].item() + offset)
                h_min = round(pred_boxes[i][1].item() - offset)
                h_max = round(pred_boxes[i][3].item() + offset)
                if class_name not in ['图片', '表格', '公式']:
                    new_img, upper_bound, lower_bound = remove_broken_lines(
                        img[h_min:h_max, w_min:w_max, :]
                    )
                    target_img_box_dicts.append({
                        'image': Image.fromarray(new_img),
                        'bbox': {
                            'x': w_min,
                            'y': h_min+upper_bound,
                            'width': w_max-w_min,
                            'height': lower_bound-upper_bound,
                            'label': class_name,
                            'score': pred_scores[i].item()
                        }
                    })
                else:
                    new_img = img[h_min:h_max, w_min:w_max, :]
                    target_img_box_dicts.append({
                        'image': Image.fromarray(new_img),
                        'bbox': {
                            'x': w_min,
                            'y': h_min,
                            'width': w_max-w_min,
                            'height': h_max-h_min,
                            'label': class_name,
                            'score': pred_scores[i].item()
                        }
                    })
            batch_target_img_box_dicts.append(target_img_box_dicts)

        batch_target_img_box_dicts = [self.deduplicate(tmp) for tmp in batch_target_img_box_dicts]

        return batch_target_img_box_dicts

    def predict_by_pdf(self, page_tuple_list, batch_size=8):
        '''
        简介
            分析每一张图片的布局，识别其中不同的目标
        输入
            n个pdf文件的所有图片
            page_tuple_list: [
                (
                    {
                        'pdf_file': '/the/path/of.pdf',
                        'out_path': '/path/of/result',
                        'page_num': int
                    },
                    Image
                ),
                ...
            ]
        输出
            pdfs_ordered_target_images: {
                '/path/of/result': {
                    1: {
                        'head': [(image, box), ...]
                        'body': [(image, box), ...]
                        ...
                    }
                    2: {
                        'head': [(image, box), ...]
                        'body': [(image, box), ...]
                        ...
                    }
                    ...
                },
                '/another/out/path': {
                    1: {
                        'head': [(image, box), ...]
                        'body': [(image, box), ...]
                        ...
                    }
                    2: {
                        'head': [(image, box), ...]
                        'body': [(image, box), ...]
                        ...
                    }
                    ...
                },
                ...
            }
            page_img_label_list: [
                {
                    'pdf_file': '/the/path/of/pdf.pdf'
                    'out_path': '/the/path/of/pdf',
                    'page_num': int,
                    'dpi': 300,
                    'bbox': [
                        {
                            "x":273.8,
                            "y":389.0,
                            "width":1390.0,
                            "height":3024.9,
                            "label":"正文"
                        },
                        ...
                    ]
                }
            ]
        存储路径
            /pdf
                -attention.pdf
                /attention
                    1.json
                    1.png
                    /1_target
                        正文_1.png
                        正文_2.png
                        ...
                    2.json
                    2.png
                    /2_target
                        标题_1.png
                        正文_1.png
                        正文_2.png
                        ...
                    3.json
                    3.png
                    /3_target
                        标题_1.png
                        作者_1.png
                        作者_2.png
                        ...
                    ...
        '''
        # 获取版面分析的结果
        batch_output_list = []
        pdfs_ordered_target_images = defaultdict(dict)
        pdfs_bboxes = defaultdict(dict)
        page_dict_list = [tmp[0] for tmp in page_tuple_list]
        page_img_list = [np.array(tmp[1]) for tmp in page_tuple_list]
        # page_img_list = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in page_img_list]
        
        start = time()
        for i in tqdm(
            range(0, len(page_img_list), batch_size), desc='rcnn', ncols=60
        ):
            batch_output_list.append(self.layout_predictor(page_img_list[i:i+batch_size]))
        print('420 rcnn 时间', time()-start)
        
        start = time()
        for idx, pointer in enumerate(tqdm(
            range(0, len(page_dict_list), batch_size), desc='parse', ncols=60
        )):
            # start = time()
            batch_target_img_box_dicts = self.parse_batch_pdf_image(
                page_img_list[pointer:pointer+batch_size], batch_output_list[idx]
            )
            # print('436 第一步解析时间', time()-start)
            # 对分析结果进行排序
            # start = time()
            for i, (page_dict, target_img_box_dicts) in enumerate(zip(page_dict_list[pointer:pointer+batch_size], batch_target_img_box_dicts)):
                # print(422, page_dict['page_num'])
                page_bboxes = [target_dict['bbox'] for target_dict in target_img_box_dicts]
                page_target_images = [target_dict['image'] for target_dict in target_img_box_dicts]
                pdfs_bboxes[page_dict['out_path']][int(page_dict['page_num'])] = page_bboxes
                # 将元素按照阅读顺序排序
                sorted_element_dic = self.sort_layout.sort_to_reading_order_by_cls(page_target_images, page_bboxes)
                pdfs_ordered_target_images[page_dict['out_path']][int(page_dict['page_num'])] = sorted_element_dic

            # print('453 第二步解析时间', time()-start)
        print('454 结果解析时间', time()-start)

        return pdfs_bboxes, pdfs_ordered_target_images

    def predict(self, image_list):
        '''
        简介
            分析每一张图片的布局，识别其中不同的目标
        输入
            n个pdf文件的所有图片
            image_list: [
                Image, Image, ...
            ]
        输出
            layout_of_img_list: [
                [
                    {
                        "x":573.8,
                        "y":389.0,
                        "width":1540.0,
                        "height":224.9,
                        "column":0,
                        "label":"标题"
                    },
                    {
                        "x":273.8,
                        "y":698.0,
                        "width":1390.0,
                        "height":3024.9,
                        "column":0,
                        "label":"正文"
                    },
                    ...
                ],
                ...
            ]
            target_img_of_img_list: [
                [img1, img2, ...],
                ...
            ]
        '''
        # 获取版面分析的结果
        image_list = [np.array(tmp) for tmp in image_list]
        layout_of_img_list = []
        target_img_of_img_list = []
        output_list = self.layout_predictor(image_list)
        batch_target_img_box_list = self.parse_batch_pdf_image(image_list, output_list)
        for target_img_box_list in batch_target_img_box_list:
            bbox_list = [target_dict['bbox'] for target_dict in target_img_box_list]
            target_image_list = [target_dict['image'] for target_dict in target_img_box_list]
            sorted_result = self.sort_layout.sort_to_reading_order(
                target_image_list, bbox_list
            )
            layout_of_img_list.append([tmp[1] for tmp in sorted_result])
            target_img_of_img_list.append([tmp[0] for tmp in sorted_result])
        return layout_of_img_list, target_img_of_img_list



