import os
from itertools import chain
from module_layout_parser import LayoutParser
from utils import get_free_gpu
from reader_writer import read_json

class LayoutParserApi:
    def __init__(
        config_file = None,     # 配置文件所在位置
        model_path = None,      # 模型权重所在位置
        idx2name_path = None,   # 序号到类别名称的映射文件
        device = None           # 用哪个设备进行运算
    ):
        # 获取该文件所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if not config_file:
            config_file = os.path.join(current_dir, 'sjt_configs/cascade/cascade_dit_base.yaml')
        if not model_path:
            model_path = os.environ.get(
                'LAYOUT_PARSER_MODEL_PATH',
                os.path.join(current_dir, 'model_final.pth')
            )
        if not idx2name_path:
            idx2name_path = os.path.join(current_dir, 'cls_idx2name.json')
        if not device:
            device_num = get_free_gpu()     # 获取空闲GPU
            device = f"cuda:{device_num}" if device_num else "cpu"
        
        idx2cls = read_json(idx2name_path)
        idx2cls = {idx: value['name'] for idx, value in idx2cls.items()}
        self.layout_parser = LayoutParser(config_file, model_path, idx2cls, device)
    
    def predict(self, image_list):
        layout_of_img_list, target_img_of_img_list = self.layout_parser.predict(
            image_list
        )
        return layout_of_img_list, target_img_of_img_list

if __name__=='__main__':
    from utils import get_page_dict_from_pdf
    layout_parser_api = LayoutParserApi()
    
    pdf_file_list = [
        '../pdf_parser/pdf/Chinese NER Using Lattice LSTM.pdf'
    ]
    target_dpi = 400
    page_tuple_list = [get_page_dict_from_pdf(pdf_file, target_dpi) for pdf_file in pdf_file_list]
    page_tuple_list = list(chain(*page_tuple_list))
    image_list = [tmp[1] for tmp in page_tuple_list]

    layout_of_img_list, target_img_of_img_list = layout_parser_api.predict(
        image_list
    )
    for layout_of_page in layout_of_img_list:
        print(layout_of_page)