import re
import cv2
import os
import fitz
import subprocess
import numpy as np
from PIL import Image
from PIL import ImageDraw, ImageFont

def get_free_gpu():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,nounits,noheader'], stdout=subprocess.PIPE)
    gpus = result.stdout.decode().strip().split('\n')
    gpus = [gpu.split(', ') for gpu in gpus]
    free_gpus = sorted(gpus, key=lambda x: int(x[1]), reverse=True)
    if int(free_gpus[0][1])<1800: return None
    return free_gpus[0][0] if free_gpus else None

def get_page_dict_from_pdf(pdf_file, target_dpi=300, debug=False):
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

def draw_bounding_box(image, boxes, output_path, color=(0, 255, 0), thickness=2):
    top_lefts, bottom_rights = [], []
    for box in boxes:
        top_lefts.append((box[0], box[1]))
        bottom_rights.append((box[0]+box[2], box[1]+box[3]))
    # 画矩形框
    for top_left, bottom_right in zip(top_lefts, bottom_rights):
        cv2.rectangle(image, top_left, bottom_right, color, thickness)

    # 保存结果
    cv2.imwrite(output_path, image)

def resize_image(img, scale):
    # 获取原始尺寸
    original_width, original_height = img.size
    
    # 计算新的尺寸
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # 调整图像大小
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)  # 使用高质量的缩放过滤器
    
    return resized_img

def detect_text_height(image):
    '''检测图片中文本的高度'''
    # 转换为灰度图
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)

    # 应用阈值操作，将图像二值化
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # 找到图像中的所有轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 遍历每个轮廓，计算高度
    boxes = [cv2.boundingRect(contour) for contour in contours]
    heights = [box[-1] for box in boxes]
    # mean_height = sum(heights)/len(heights)
    # heights = [box[-1] for box in boxes if box[-1]>mean_height]

    return int(sum(heights)/len(heights)*3)

from PIL import Image, ImageDraw, ImageFont

def create_image_with_text(text, image_width=500, image_height=500):
    # 创建一个新图像，背景为白色
    image = Image.new('RGB', (image_width, image_height), 'white')
    draw = ImageDraw.Draw(image)
    
    # 使用Pillow库的默认字体，或者替换为你自己的.ttf文件路径
    try:
        # 尝试加载一个.ttf字体文件
        font = ImageFont.truetype("arial.ttf", 20)  # 你可能需要调整字体和字号
    except IOError:
        # 如果无法加载字体文件，则使用默认字体
        font = ImageFont.load_default()
    
    # 文本换行处理
    margin = 10
    max_width = image_width - 2 * margin
    max_height = image_height - 2 * margin
    lines = []
    words = text.split()
    line = words.pop(0)

    for word in words:
        test_line = line + ' ' + word
        width, _ = font.textsize(test_line, font=font)
        if width <= max_width:
            line = test_line
        else:
            lines.append(line)
            line = word
    lines.append(line)  # 添加最后一行

    # 计算文本块的总高度
    total_text_height = 0
    line_heights = []
    for line in lines:
        _, line_height = font.textsize(line, font=font)
        line_heights.append(line_height)
        total_text_height += line_height
    
    # 计算文本块开始的y位置，以便垂直居中
    y = (image_height - total_text_height) // 2

    # 在图像上绘制文本
    for i, line in enumerate(lines):
        width, _ = font.textsize(line, font=font)
        x = (image_width - width) // 2  # 计算x位置，以便水平居中
        draw.text((x, y), line, fill="black", font=font)
        y += line_heights[i]  # 移动到下一行

    return image
    # 保存或显示图像
    # image.show()
    # image.save('output.png')  # 保存图片



if __name__=='__main__':
    base_path = '/home/ninziwei/projects/pdf_parser/pdf'
    
    '''测试图像缩放函数'''
    # image_path = f'{base_path}/2202.02506v1/3.png'
    # img = Image.open(image_path)
    # scale = 0.667
    # resized_img = resize_image(img, scale)
    # resized_img.save(image_path[:-4]+f'-{scale}.png')

    font = ImageFont.load_default()
    print(font.getbbox('sdfgr dsfg sd'))

    '''测试文本生成函数'''
    # # 例子使用的文本
    # text = "这里是一个包含500字的示例文本，这段文本将会被自动分割成多行，确保每行都能适应图像的宽度，并且文本在图像中垂直和水平居中。" * 10
    # # 创建图像
    # new_image = create_image_with_text(text, 350, 800)
    # new_image.save('./new.png')