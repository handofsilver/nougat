import os
import fitz
import json
import pickle
from PIL import Image, ImageFont

def read_list(file_path):
    with open(file_path, 'rb') as file:
        my_list = pickle.load(file)
    return my_list

def write_list(file_path, my_list):
    with open(file_path, 'wb') as file:
        pickle.dump(my_list, file)

def read_json(file_path):
    '''读取json格式的一体存储的文件'''
    with open(file_path, 'r', encoding='utf8') as f:
        data_list = json.loads(f.read())
    return data_list

def write_json_to_break(path, new_dict):
    '''将json文件转写成可以换行的易读格式'''
    data = json.dumps(new_dict, indent=1, ensure_ascii=False)
    with open(path, 'w', newline='\n') as f:
        f.write(data)

def read_txt(file_path):
    '''读取纯文本文件'''
    with open(file_path, 'r', encoding='utf8') as f:
        text = f.read()
    return text.strip()

def write_txt(file_path, data):
    '''存储txt文件'''
    with open(file_path, 'w', encoding='utf8') as f:
        f.write(data)

def read_page_dict_from_pdf(pdf_file, target_dpi=300, debug=False):
    '''
    输入
        pdf_file: '/the/path/of.pdf'
        target_dpi: 指定目标分辨率（以点/英寸为单位）
        debug: 调试格式下才需要存储原始图像
    输出
        page_img_list: [
            (
                {
                    'pdf_file': '/the/path/of.pdf',
                    'out_path': '/the/path/of',
                    'page_num': int,
                    'image': Image
                },
                Image
            )
            ...
        ]
    '''
    out_path = pdf_file[:-4]
    if not os.path.exists(out_path) and debug:
        os.mkdir(out_path)

    # 打开PDF文件
    pdf_document = fitz.open(pdf_file)
    
    # 转换PDF页面为图像
    page_tuple_list = []
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        # 计算缩放因子，以实现目标分辨率
        zoom_x = target_dpi / 72  # 72 points = 1 inch
        zoom_y = target_dpi / 72
        # 获取缩放后的Pixmap对象
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom_x, zoom_y))
        # print('reader_writer 72', page_num)
        page_tuple_list.append((
            {
                'pdf_file': pdf_file,
                'out_path': out_path,
                'page_num': page_num,
            },
            Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        ))
        # 保存图像
        if debug and out_path:
            image_path = os.path.join(out_path, f'{page_num+1}.png')
            pix.save(image_path)
    # 关闭PDF文件
    pdf_document.close()
    return page_tuple_list