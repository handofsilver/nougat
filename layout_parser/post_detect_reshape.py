'''
将多个目标拼接成高度接近最长目标的大图片，然后送入nougat-ocr识别

'''
import cv2
import numpy as np
from PIL import Image, ImageDraw
from utils import draw_bounding_box


# 将图片分割为多个小图，保证能把目标尽可能拼到一张大图上
def find_split_points(projection, num_splits=3):
    # 找到没有文字的部分作为分割点
    total_height = len(projection)
    segment_height = total_height // num_splits
    split_points = []

    for i in range(1, num_splits):
        start = max(i * segment_height - 30, 0)
        end = i * segment_height + segment_height // 5
        segment = projection[start:end]
        split_point = start + np.argmin(segment)
        split_points.append(split_point)
    
    return split_points

def split_image(image, num_splits=3):
    # 读取图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 二值化
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    projection = np.sum(binary, axis=1)

    split_points = find_split_points(projection, num_splits)

    # 根据分割点将图像分成 n 段
    start, segments = 0, []
    for point in split_points:
        segments.append(image[start:point, :])
        start = point
    segments.append(image[start:, :])
    
    return segments

def split_image_tuple(image_label_tuple, max_height):
    '''
    image_label_tuple: (image, 'class name', x_gap)
    '''
    # 规范化输入
    new_image_label_tuples = []
    if image_label_tuple and len(image_label_tuple) == 2:
        image_label_tuple = (image_label_tuple[0], image_label_tuple[1], 0)

    image, label, x_gap = image_label_tuple
    if label=='公式':
        return [image_label_tuple]
    # 判断将图片分割成几个小图
    if image.size[1]>0.9*max_height:
        splited_num = 20
    if image.size[1]>0.8*max_height:
        splited_num = 16
    elif image.size[1]>0.6*max_height:
        splited_num = 12
    elif image.size[1]>0.4*max_height:
        splited_num = 9
    elif image.size[1]>0.3*max_height:
        splited_num = 7
    elif image.size[1]>0.2*max_height:
        splited_num = 5
    elif image.size[1]>0.12*max_height:
        splited_num = 3
    elif image.size[1]>0.08*max_height:
        splited_num = 2
    else:
        splited_num = 1
    # 分割图片
    if splited_num>1:
        segments = split_image(np.array(image), splited_num)
        new_image_label_tuples += [(Image.fromarray(seg), label, x_gap) for seg in segments]
    else:
        new_image_label_tuples.append((image, label, x_gap))
    return new_image_label_tuples

def split_image_tuples(body_image_label_tuples, max_height):
    '''
    将多个tuple切分成多个小块
    body_image_label_tuples:
    [
        (image, 'class name', x_gap),
        (image, 'class name', x_gap),
        ...
    ]
    '''
    if not body_image_label_tuples: 
        return []
    new_body_image_label_tuples = []
    splited_nums = []
    for image_label_tuple in body_image_label_tuples:
        new_body_image_label_tuples += split_image_tuple(image_label_tuple, max_height)
    return new_body_image_label_tuples


# 将图片裁剪为一张小图
def crop_to_top_img(image):
    '''
    裁剪锚点图片来减少工作量，从而提高ocr效率
    保留最上面的部分，至少留两行
    '''
    height = min(max(image.size[0]//6, 80), image.size[1])
    image = Image.fromarray(np.array(image)[:height,:])
    return image

def crop_to_bottom_img(image):
    '''
    裁剪锚点图片来减少工作量，从而提高ocr效率
    保留最下面的部分，至少留两行
    '''
    height = min(max(image.size[0]//6, 80), image.size[1])
    image = Image.fromarray(np.array(image)[image.size[1]-height:,:])
    return image

def remove_broken_lines(image):
    '''
    接受 np.array 格式的图片
    自适应地裁剪掉所有被截断的行
    '''
    # 读取图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 二值化
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 寻找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 找到图像的高度
    img_height = image.shape[0]
    
    # 过滤掉过小的轮廓并确定裁剪区域
    boxes = [cv2.boundingRect(contour) for contour in contours]
    heights = [box[-1] for box in boxes]
    min_contour_height = int(sum(heights)/len(heights)*1.2)  # 根据需要调整最小轮廓高度
    y_coords = []
    for _, y, _, h in boxes:
        if h >= min_contour_height:
            y_coords.append(y)
            y_coords.append(y + h)
    
    # 确定上边界和下边界
    if y_coords:
        upper_bound = max(min(y_coords)-3, 0)
        lower_bound = min(max(y_coords)+3, img_height)
    else:
        upper_bound = 0
        lower_bound = img_height
    
    # 裁剪图像
    cropped_image = image[upper_bound:lower_bound, :]
    
    return cropped_image, upper_bound, lower_bound

def crop_left_right_borders(image):
    '''裁剪掉图像左右两边的白边'''
    # 转换为灰度图像
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 二值化处理，假设白色为背景色
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # 找到所有非零（非白色）像素的列索引
    cols = np.any(binary > 0, axis=0)
    left_border = np.argmax(cols)
    right_border = binary.shape[1] - np.argmax(cols[::-1])

    # 裁剪图像
    cropped_image = image[:, left_border:right_border]
    return Image.fromarray(cropped_image)

def crop_top_bottom_borders(image):
    '''裁剪掉图像上下两边的白边'''
    # 转换为灰度图像
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 二值化处理，假设白色为背景色
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # 找到所有非零（非白色）像素的列索引
    cols = np.any(binary > 0, axis=1)
    top_border = np.argmax(cols)
    bottom_border = binary.shape[1] - np.argmax(cols[::-1])

    # 裁剪图像
    cropped_image = image[top_border:bottom_border, :]
    return Image.fromarray(cropped_image)

def remove_last_two_words(image, kernel_size):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 二值化处理
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # 膨胀操作
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    
    # 寻找轮廓
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 获取所有轮廓的边界框
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]

    # 按照Y坐标排序，找到最后一行
    bounding_boxes = sorted(bounding_boxes, key=lambda x: x[1], reverse=True)
    thresh = np.mean([box[1] for box in bounding_boxes[:5]])
    last_line_boxes = [box for box in bounding_boxes if np.abs(box[1]-thresh)<10]
    
    # 按照X坐标排序，找到最后两个单词
    last_line_boxes = sorted(last_line_boxes, key=lambda x: x[0])
    if len(last_line_boxes) >= 2:
        # 删除最后两个单词
        for box in last_line_boxes[-2:]:
            print(box)
            x, y, w, h = box
            image[y:y+h, x:x+w] = 255  # 置为白色
    
    return image

def crop_broken_line(image):
    '''
    保留 crop_position 以上的部分
    '''
    mean_line_height, boxes = detect_text_height(image)
    first_line_heights = [box[-1] for box in boxes[-10:] if box[-1]>3]
    last_line_heights = [box[-1] for box in boxes[:10] if box[-1]>3]
    start_crop_position, end_crop_position = 0, -1
    if sum(first_line_heights)/len(first_line_heights)<0.6*mean_line_height:
        start_crop_position = max([box[1]+box[3] for box in boxes[-3:]]) + 2
    if sum(last_line_heights)/len(last_line_heights)<0.6*mean_line_height:
        end_crop_position = min([box[1] for box in boxes[:3]])
    print('img_reshape 39', start_crop_position, end_crop_position)
    image = Image.fromarray(np.array(image)[start_crop_position:end_crop_position,:])
    return image

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

def add_period(image, x_box, y_boxes, mean_height):
    '''
    简介
        在某一行的结尾添加句号图像
    输入
        image: np.array
        x_box: 句号的x坐标
        y_box: 句号所在行的所有y坐标
        mean_height: 字符的平均高度
    '''
    x_tail = x_box[0] + x_box[2]
    all_y = [box[1]+box[3] for box in y_boxes if box[3]>mean_height]
    y_tail = int(sum(all_y)/len(all_y))
    point_size = int(mean_height//3.5)
    image[
        y_tail-point_size+2:y_tail+2, 
        x_tail+point_size-1:x_tail+point_size*2-1
    ] = 0
    return image

def find_last_line_boxes(image):
    '''找到图片的最后一行的所有字符'''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 二值化处理，假设白色为背景色
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    '''找到文本高度'''
    # 找到图像中的所有轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 遍历每个轮廓，计算高度
    boxes = [cv2.boundingRect(contour) for contour in contours]
    heights = [box[-1] for box in boxes]
    mean_height = sum(heights)/len(heights)
    line_height = int(mean_height*2)
    
    '''找到最后一行'''
    boxes = sorted(boxes, key=lambda x: x[1])
    big_boxes = [box for box in boxes if box[3]>mean_height]
    thresh = line_height/1.2
    center = big_boxes[-1][1]
    last_line_boxes = [box for box in boxes if abs(box[1]-center)<thresh and box[1]+box[3]>center]
    last_line_boxes = sorted(last_line_boxes, key=lambda box: box[0]+box[2])

    return last_line_boxes, mean_height

''' 对单独的目标图片进行一系列处理 '''
def deal_body_images(image_label_gap_tuples):
    '''判断是否需要加句号'''
    pre_label = ''
    for i, (image, label, x_gap) in enumerate(image_label_gap_tuples[:-1]):
        # print(342, label)
        next_label = image_label_gap_tuples[i+1][1]
        if not (label=='正文' and next_label=='子标题'):
            pre_label = label
            continue
        if not (image.size[0]/image.size[1]>10):
            pre_label = label
            continue
        
        image = np.array(image)
        last_line_boxes, mean_height = find_last_line_boxes(image)
        x, y, w, h = last_line_boxes[-1]
        if h>mean_height/1.2:
            image = add_period(image, last_line_boxes[-1], last_line_boxes, mean_height)
            # image_label_gap_tuples[i-1][0].save('xx1.png')
            # image_label_gap_tuples[i][0].save('xx2.png')
            # image_label_gap_tuples[i+1][0].save('xx3.png')
            if [pre_label, label, next_label]==['子标题', '正文', '子标题']:
                label = '不在最后'
                # print(355)
            image_label_gap_tuples[i] = (Image.fromarray(image), label, x_gap)
        
        pre_label = label
        # draw_bounding_box(
        #     image.copy(), 
        #     last_line_boxes, 
        #     './body_box.png'
        # )
    return image_label_gap_tuples

''' 对end_main进行裁剪、删掉头尾单词、补充句号 '''
def deal_end_main_image(image, standard_line_height, img_num=0, debug=False):
    '''
    简介
        删掉第一行的第一个单词
    输入
        image: 原始图像
        mean_height: 将图像缩放到 mean_height
    '''
    # 转换为灰度图像
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 二值化处理，假设白色为背景色
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    '''找到文本高度'''
    # 找到图像中的所有轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 遍历每个轮廓，计算高度
    boxes = [cv2.boundingRect(contour) for contour in contours]
    heights = [box[-1] for box in boxes]
    mean_height = sum(heights)/len(heights)
    line_height = int(mean_height*2)

    '''裁剪掉左右白边'''
    # # 找到所有非零（非白色）像素的列索引
    # cols = np.any(binary > 0, axis=0)
    # left_border = np.argmax(cols)
    # right_border = binary.shape[1] - np.argmax(cols[::-1])
    # # 裁剪图像
    # cropped_image = image[:, left_border:right_border]
    # cropped_binary = binary[:, left_border:right_border]

    '''裁剪掉上下白边'''
    # # 找到所有非零（非白色）像素的列索引
    # cols = np.any(cropped_binary > 0, axis=1)
    # top_border = np.argmax(cols)
    # bottom_border = cropped_binary.shape[1] - np.argmax(cols[::-1])
    # # 裁剪图像
    # cropped_image = cropped_image[top_border:bottom_border, :]
    # cropped_binary = cropped_binary[top_border:bottom_border, :]

    cropped_image = image
    cropped_binary = binary

    # 在上下位置补充白边
    border = np.ones((6, cropped_image.shape[1], 3), dtype=np.uint8) * 255
    cropped_image = np.vstack([border, cropped_image, border])
    border = np.ones((cropped_image.shape[0], line_height//5, 3), dtype=np.uint8) * 255
    cropped_image = np.hstack([border, cropped_image, border])
    border = np.ones((5, cropped_binary.shape[1]), dtype=np.uint8) * 0
    cropped_binary = np.vstack([border, cropped_binary, border])
    border = np.ones((cropped_binary.shape[0], line_height//5), dtype=np.uint8) * 0
    cropped_binary = np.hstack([border, cropped_binary, border])

    '''获取单词的边界'''
    # 膨胀操作
    kernel = np.ones((max(2, round(line_height/12)), round(line_height/4.2)), np.uint8)
    dilated = cv2.dilate(cropped_binary, kernel, iterations=1)
    # 寻找轮廓
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 获取所有轮廓的边界框
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    # 按照Y坐标从大到小排序
    bounding_boxes = sorted(bounding_boxes, key=lambda x: x[1], reverse=True)
    expan_heights = [box[3] for box in bounding_boxes]
    mean_expan_height = round(sum(expan_heights)/len(expan_heights))-2
    big_expan_boxes = [box for box in bounding_boxes if box[3]>mean_expan_height]

    '''删掉图片头尾的几个单词'''
    # 找到最后一行
    # center = np.mean([box[1] for box in bounding_boxes[:5]])
    center = big_expan_boxes[0][1]
    thresh = round(line_height/1.2)
    last_line_boxes = [box for box in bounding_boxes if np.abs(box[1]-center)<thresh and box[1]+box[3]>center]
    # 按照X坐标从小到大排序
    last_line_boxes = sorted(last_line_boxes, key=lambda x: x[0]*10+1/x[2])
    last_line_boxes = merge_point_of_ij(last_line_boxes)
    # 删除最后三个单词（大于两行且最后一行大于三个词且右边足够靠近边界）
    close_enough = (cropped_image.shape[0]-(last_line_boxes[-1][0]+last_line_boxes[-1][2]))<line_height*2
    flag1 = cropped_binary.shape[0]>2*line_height and len(last_line_boxes)>3 and close_enough
    flag2 = cropped_binary.shape[0]<2*line_height and len(last_line_boxes)>7
    if flag1 or flag2:
    # if len(last_line_boxes)>5 and close_enough:
        for box in last_line_boxes[-3:]:
            x, y, w, h = box
            cropped_image[y:y+h+2, x:x+w] = 255  # 置为白色
        # 添加句号
        cropped_image = add_period(cropped_image, last_line_boxes[-4], last_line_boxes, mean_height)
    else:
        # 添加句号
        cropped_image = add_period(cropped_image, last_line_boxes[-1], last_line_boxes, mean_height)

    # 找到第一行
    # center = np.mean([box[1] for box in bounding_boxes[:5]])
    center = big_expan_boxes[-1][1]
    first_line_boxes = [box for box in bounding_boxes if np.abs(box[1]-center)<thresh and box[1]+box[3]>center]
    # 按照X坐标从小到大排序
    first_line_boxes = sorted(first_line_boxes, key=lambda x: x[0]*10+1/x[2])
    first_line_boxes = merge_point_of_ij(first_line_boxes)
    # 删除第一个单词（第一行左边足够靠近边界且大于两行且第一个单词足够短）
    # print(413, first_line_boxes[0][0]<line_height and cropped_binary.shape[0]>2*line_height and first_line_boxes[0][2]<4*mean_expan_height)
    # print(first_line_boxes[0][0], mean_expan_height, line_height)
    # print(first_line_boxes[0][2], 4*line_height)
    if first_line_boxes[0][0]<line_height and cropped_binary.shape[0]>2*line_height and first_line_boxes[0][2]<4*mean_expan_height:
        for box in first_line_boxes[:1]:
            x, y, w, h = box
            cropped_image[max(y-2,0):y+h+2, max(x-2,0):x+w] = 255  # 置为白色
    
    # 存储 end_main 的边界框
    if debug:
        draw_bounding_box(
            cropped_image.copy(), 
            first_line_boxes+last_line_boxes, 
            f'./debug/end_main_{img_num}_box.png'
        )

    '''缩放图片'''
    # 计算新的尺寸
    scale_factor = standard_line_height/line_height
    # print(328, scale_factor)
    height, width = cropped_image.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    # print(330, height, width)
    # print(330, new_height, new_width)
    # 调整图像大小
    resized_image = cv2.resize(cropped_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    return Image.fromarray(resized_image)

def deal_minor_image(image, standard_line_height, class_name, img_num=0, debug=False):
    '''
    简介
        删掉第一行的第一个单词
    输入
        image: 原始图像
        mean_height: 将图像缩放到 mean_height
    '''
    # 转换为灰度图像
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 二值化处理，假设白色为背景色
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    '''裁剪掉左右白边'''
    # 找到所有非零（非白色）像素的列索引
    # cols = np.sum(binary > 0, axis=0)
    # print(cols)
    # left_border = np.argmax(cols)
    # right_border = binary.shape[1] - np.argmax(cols[::-1])
    # print(left_border, right_border)
    # # 裁剪图像
    # cropped_image = image[:, left_border:right_border]
    # cropped_binary = binary[:, left_border:right_border]
    # for x in cropped_binary:
    #     print(505, x)

    '''找到文本高度'''
    # 找到图像中的所有轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 遍历每个轮廓，计算高度
    boxes = [cv2.boundingRect(contour) for contour in contours]
    heights = [box[-1] for box in boxes]
    mean_height = sum(heights)/len(heights)
    line_height = int(mean_height*2)
    
    '''裁剪掉上下白边'''
    # # 找到所有非255（非白色）像素的行索引
    # cols = np.sum(cropped_binary > 0, axis=1)
    # top_border = np.argmax(cols)
    # bottom_border = cropped_binary.shape[1] - np.argmax(cols[::-1])
    # # 裁剪图像
    # cropped_image = cropped_image[top_border:bottom_border, :]
    # cropped_binary = cropped_binary[top_border:bottom_border, :]
    # for x in cropped_binary:
    #     print(524, x)

    cropped_image = image
    cropped_binary = binary
    # 在上下左右位置补充白边
    border = np.ones((6, cropped_image.shape[1], 3), dtype=np.uint8) * 255
    cropped_image = np.vstack([border, cropped_image, border])
    border = np.ones((cropped_image.shape[0], line_height//5, 3), dtype=np.uint8) * 255
    cropped_image = np.hstack([border, cropped_image, border])
    border = np.ones((5, cropped_binary.shape[1]), dtype=np.uint8) * 0
    cropped_binary = np.vstack([border, cropped_binary, border])
    border = np.ones((cropped_binary.shape[0], line_height//5), dtype=np.uint8) * 0
    cropped_binary = np.hstack([border, cropped_binary, border])

    '''获取单词的边界'''
    # 膨胀操作
    # kernel = np.ones((max(2, round(line_height/12)), round(line_height/4.2)), np.uint8)
    kernel_size = max(2, round(line_height/20))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(cropped_binary, kernel, iterations=1)
    # 寻找轮廓
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 获取所有轮廓的边界框
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    # 按照Y坐标从大到小排序
    bounding_boxes = sorted(bounding_boxes, key=lambda x: x[1], reverse=True)
    expan_heights = [box[3] for box in bounding_boxes]
    mean_expan_height = round(sum(expan_heights)/len(expan_heights))-2
    big_expan_boxes = [box for box in bounding_boxes if box[3]>mean_expan_height]

    # if debug:
    #     draw_bounding_box(
    #         cropped_image.copy(), 
    #         bounding_boxes, 
    #         f'./debug/minor_{img_num}_box.png'
    #     )

    '''删除掉第一个字符'''
    if class_name=='footnote':
        # 找到第一行
        thresh = round(line_height/1.2)
        center = big_expan_boxes[-1][1]
        first_line_boxes = [box for box in bounding_boxes if np.abs(box[1]-center)<thresh and box[1]+box[3]>center]
        # 按照X坐标从小到大排序
        first_line_boxes = sorted(first_line_boxes, key=lambda x: x[0]*10+1/x[2])
        first_line_boxes = merge_point_of_ij(first_line_boxes)
        # 删除第一个单词（第一行左边足够靠近边界且大于两行且第一个单词足够短）
        for box in first_line_boxes[:1]:
            x, y, w, h = box
            cropped_image[max(y-2,0):y+h+2, max(x-2,0):x+w] = 255  # 置为白色
        
        # 存储 end_main 的边界框
        if debug:
            draw_bounding_box(
                cropped_image.copy(), 
                first_line_boxes, 
                f'./debug/minor_{img_num}_box.png'
            )

    '''添加句号'''
    # 找到最后一行
    center = big_expan_boxes[0][1]
    thresh = round(line_height/1.2)
    last_line_boxes = [box for box in bounding_boxes if np.abs(box[1]-center)<thresh and box[1]+box[3]>center]
    # 按照X坐标从小到大排序
    last_line_boxes = sorted(last_line_boxes, key=lambda x: x[0]*10+1/x[2])
    last_line_boxes = merge_point_of_ij(last_line_boxes)
    # 添加句号
    cropped_image = add_period(cropped_image, last_line_boxes[-1], last_line_boxes, mean_height)

    '''缩放图片'''
    # 计算新的尺寸
    scale_factor = standard_line_height/line_height
    height, width = cropped_image.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    # 调整图像大小
    resized_image = cv2.resize(cropped_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    return Image.fromarray(resized_image)

def deal_footnote_image(image, img_num=0):
    '''
    简介
        删掉第一行的第一个单词
    输入
        image: 原始图像
        mean_height: 将图像缩放到 mean_height
    '''
    # 转换为灰度图像
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 二值化处理，假设白色为背景色
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    '''找到文本高度'''
    # 找到图像中的所有轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 遍历每个轮廓，计算高度
    boxes = [cv2.boundingRect(contour) for contour in contours]
    heights = [box[-1] for box in boxes]
    max_height, min_height = max(heights), max(heights)/6
    mean_height = sum(heights)/len(heights)
    line_height = int(mean_height*2)


    '''删掉图片头尾的几个单词'''
    # 膨胀操作
    kernel = np.ones((max(2, round(line_height/12)), round(line_height/4.2)), np.uint8)
    dilated = cv2.dilate(cropped_binary, kernel, iterations=1)
    # 寻找轮廓
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 获取所有轮廓的边界框
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    # 按照Y坐标从大到小排序
    bounding_boxes = sorted(bounding_boxes, key=lambda x: x[1], reverse=True)
    expan_heights = [box[3] for box in bounding_boxes]
    mean_expan_height = round(sum(expan_heights)/len(expan_heights))-2
    big_expan_boxes = [box for box in bounding_boxes if box[3]>mean_expan_height]

    # 找到第一行
    thresh = round(line_height/1.2)
    center = big_expan_boxes[-1][1]
    first_line_boxes = [box for box in bounding_boxes if np.abs(box[1]-center)<thresh and box[1]+box[3]>center]
    # 按照X坐标从小到大排序
    first_line_boxes = sorted(first_line_boxes, key=lambda x: x[0]*10+1/x[2])
    first_line_boxes = merge_point_of_ij(first_line_boxes)
    # 删除第一个单词（第一行左边足够靠近边界且大于两行且第一个单词足够短）
    for box in first_line_boxes[:1]:
        x, y, w, h = box
        cropped_image[max(y-2,0):y+h+2, max(x-2,0):x+w] = 255  # 置为白色
    
    # draw_bounding_box(
    #     cropped_image.copy(), 
    #     bounding_boxes, 
    #     f'./footnote_{img_num}_box.png'
    # )

    return cropped_image

def merge_point_of_ij(boxes):
    '''
    简介
        同一行的目标中，i、j等字母的点可能没有通过膨胀操作合并
        通过本函数将字母上的点正确地合并到边界框内
    '''
    new_boxes = [boxes[0]]
    for box in boxes[1:]:
        if box[0]>=new_boxes[-1][0] and box[0]+box[2]<=new_boxes[-1][0]+new_boxes[-1][2]:
            new_y = min(box[1], new_boxes[-1][1])
            new_h = max(box[1]+box[3], new_boxes[-1][1]+new_boxes[-1][3]) - new_y
            new_boxes[-1] = (new_boxes[-1][0], new_y, new_boxes[-1][2], new_h)
        else:
            new_boxes.append(box)
    return new_boxes



if __name__=='__main__':
    # # Mocking the process as we don't have actual images
    # # Assuming the sizes of images for demonstration
    # image_sizes = [(200, 900, '正文'), (200, 300, '正文'), (400, 500, '公式'), (250, 350, '正文')]

    # # Creating dummy images for demonstration
    # images = [(label, Image.new('RGB', (width, height), color='blue' if label == '正文' else 'green')) for width, height, label in image_sizes]
    # max_body_width = 600

    # # Aligning images vertically as per instructions
    # long_images = vertical_align(images)
    # # Horizontally aligning long_images as per instructions
    # big_images = horizontal_align(long_images, max_body_width)

    # # Displaying the big images for verification
    # for img in big_images:
    #     img.show()

    # ''' test deal_end_main_image'''
    # for name in ['xx']:
    #     img_path = f'./{name}.png'
    #     image = Image.open(img_path).convert("RGB")
    #     out_image = deal_end_main_image(image, 50)
    #     out_image.save(f'./{name}xx.png')

    # '''test end_with_hyphen'''
    # for name in ['end_with_hyphen1']:
    #     img_path = f'./{name}.png'
    #     image = Image.open(img_path).convert("RGB")
    #     print(end_with_hyphen(image))
    
    # '''test deal_body_images'''
    # img_path = 'body_6.png'
    # image_label_gap_tuples = [(Image.open(img_path).convert("RGB"), '正文', 0), (None, '子标题', 0)]
    # image_label_gap_tuples = deal_body_images(image_label_gap_tuples)
    # image_label_gap_tuples[0][0].save('./debug/body_.png')

    '''test deal_minor_image'''
    img_path = './debug/minor_914x.png'
    image = Image.open(img_path).convert("RGB")
    deal_minor_image(image, 20, 'class_name', 914, debug=True)

    
    