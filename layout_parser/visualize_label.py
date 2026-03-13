import os
import random
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont
from reader_writer import *

random.seed(53)

def random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return (r, g, b)

def hex_to_rgb(hex_color):
    # 去除可能的'#'字符
    hex_color = hex_color.lstrip('#')

    # 拆分并转换为整数
    r, g, b = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:], 16)
    
    return [r, g, b]

def parse_annotation_from_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text.strip('\ufeff')
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        
        width = xmax - xmin
        height = ymax - ymin
        
        objects.append({
            'x': xmin,
            'y': ymin,
            'width': width,
            'height': height,
            'label': name
        })
    
    return objects


class LabelVisualizer:
    def __init__(self, idx2cls={}, base_size=30, font_path=None) -> None:
        self.idx2cls = idx2cls
        self.base_size = base_size
        self.font_path = font_path
        self._resolve_font_path()

    def _resolve_font_path(self):
        if self.font_path and os.path.isfile(self.font_path):
            return
        for candidate in ["simsun.ttc", "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"]:
            if os.path.isfile(candidate):
                self.font_path = candidate
                return
        self.font_path = None

    def _get_adaptive_font(self, image):
        if not self.font_path:
            return ImageFont.load_default()
        width, height = image.size
        reference_size = (width**2 + height**2) ** 0.5
        base_reference = (1920**2 + 1080**2) ** 0.5
        font_size = max(15, min(int(self.base_size * reference_size / base_reference), int(self.base_size * 1.5)))
        try:
            return ImageFont.truetype(self.font_path, font_size, encoding="unic")
        except (IOError, OSError):
            return ImageFont.load_default()

    def draw_label_on_image(self, image, boxes, out_path=None):
        if isinstance(image, str):
            image = Image.open(image)
        font = self._get_adaptive_font(image)
        draw = ImageDraw.Draw(image)

        for box in boxes:
            x, y = box["x"], box["y"]
            width, height = box["width"], box["height"]

            if box['label'] in self.idx2cls:
                color = self.idx2cls[box['label']].get('backgroundColor', random_color())
                text = self.idx2cls[box['label']].get('name', self.idx2cls[box['label']].get('text', box['label']))
            else:
                color = random_color()
                self.idx2cls[box['label']] = {'name': box['label'], 'backgroundColor': color}
                text = box['label']

            draw.rectangle([x, y, x + width, y + height], outline=color, width=1)
            try:
                bbox = font.getbbox(text)
                text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except AttributeError:
                text_width, text_height = font.getsize(text)
            text_x = min(x + 5, image.width - text_width - 5)
            text_y = max(y - text_height - 7, 7)
            draw.text((text_x, text_y), text, fill=color, font=font)

        if out_path:
            image.save(out_path, dpi=(200, 200))
        return image

    paint_label_on_graph = draw_label_on_image

def draw_single_img():
    idx2cls_path = 'D:/SubWork\电网\label_config.json'
    label_config = read_json(idx2cls_path)
    idx2cls = {tmp['text']:tmp for tmp in label_config}
    label_visualizer = LabelVisualizer(idx2cls)
    img_base_path = 'D:/SubWork\电网\data\归档\data-v1'
    out_base_path = 'D:/SubWork\电网\data\归档\data-v1-label'
    label_path = 'D:/SubWork\电网\data\归档\label-v1.json'
    label_list = read_json_by_line(label_path)
    
    for label in label_list:
        img_path = os.path.join(img_base_path, label['filename'])
        out_path = os.path.join(out_base_path, label['filename'])
        label_visualizer.paint_label_on_graph(img_path, label['bbox'], out_path)

def draw_from_xml():
    label_visualizer = LabelVisualizer()
    base_path = 'C:/SubWork\电网\data\\0719-样本'
    cls_name_list = os.listdir(base_path)
    for cls_name in cls_name_list:
        cls_path = os.path.join(base_path, cls_name)
        names = os.listdir(cls_path)
        names = [name[:-4] for name in names if name.endswith('.xml')]
        for name in names:
            img_path = os.path.join(cls_path, f'{name}.JPG')
            out_path = os.path.join(cls_path, f'{name}-label.JPG')
            label_path = os.path.join(cls_path, f'{name}.xml')
            boxes = parse_annotation_from_xml(label_path)
            label_visualizer.paint_label_on_graph(img_path, boxes, out_path)

def draw_pdf_images():
    label_visualizer = LabelVisualizer()
    pdf_name = '2202.02506v1'
    pdf_file_path = f'./{pdf_name}.pdf'
    out_base_path = f'./{pdf_name}'
    label_path = f'./{pdf_name}/layout.json'
    
    label_by_page = read_json(label_path)
    page_img_list = read_page_dict_from_pdf(pdf_file_path)
    page_img_list = [tmp[1] for tmp in page_img_list]

    for (img_num, boxes), page_img in zip(label_by_page.items(), page_img_list):
        out_path = os.path.join(out_base_path, f'{img_num}.png')
        label_visualizer.paint_label_on_graph(page_img, boxes, out_path)

def draw_pdf_image():
    label_visualizer = LabelVisualizer()
    pdf_name = '2202.02506v1'
    pdf_file_path = f'./{pdf_name}.pdf'
    out_base_path = f'./{pdf_name}'
    label_path = f'./{pdf_name}/layout.json'
    
    label_by_page = read_json(label_path)
    page_img_list = read_page_dict_from_pdf(pdf_file_path)
    page_img_list = [tmp[1] for tmp in page_img_list]

    for (img_num, boxes), page_img in zip(label_by_page.items(), page_img_list):
        out_path = os.path.join(out_base_path, f'{img_num}.png')
        label_visualizer.paint_label_on_graph(page_img, boxes, out_path)

if __name__=='__main__':
    # draw_single_img()

    # 示例XML文件路径
    # xml_file = 'C:/SubWork\电网\data\比武\标注大类-保险销子\\5.xml'
    # parsed_objects = parse_annotation_from_xml(xml_file)
    # for obj in parsed_objects:
    #     print(obj)

    # draw_from_xml()
    # draw_pdf_images()

    # label_visualizer = LabelVisualizer()
    # label_visualizer.paint_label_on_graph(image, boxes, out_path)

    base_path = '/home/ninziwei/projects/pdf_parser/pdf'
    pdf_file_path = f'{base_path}/1_3903682737_DSN_Parallel_Randomization_for_Large_Structured_Markov_Chains.pdf'
    pdf_file_path = f'{base_path}/2202.02506v1.pdf'
    page_img_list = read_page_dict_from_pdf(pdf_file_path, 400, True)
    print(len(page_img_list))

