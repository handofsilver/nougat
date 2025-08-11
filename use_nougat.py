'''
https://huggingface.co/docs/transformers/main/en/model_doc/nougat
https://huggingface.co/spaces/ysharma/nougat
https://github.com/facebookresearch/nougat
https://zhuanlan.zhihu.com/p/654982554
https://pypi.org/project/nougat-ocr/
'''

import re
import torch
from tqdm import tqdm
from PIL import Image
from itertools import chain
from transformers import NougatProcessor, VisionEncoderDecoderModel
#from transformers.models.nougat import NougatTokenizerFast
import os
import traceback

class NougatOCR:
    def __init__(self, model_path=None, model_type='small', device=None) -> None:
        '''
        model_path: 存储 nougat 模型的位置
        model_type: base or small
        '''
        if not model_path:
            if model_type=='base':[
                model_path = "facebook/nougat-base"
            else:
                model_path = "facebook/nougat-small"
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.processor = NougatProcessor.from_pretrained(model_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path)
        self.model.to(device)
    
    def clean_sequence(self, sequence):
        '''
        sequence: nougat-ocr的识别结果
        '''
        # 将括号转化为 $ 以确保文本能够被正确解析
        dollar_sequence = sequence.replace('\)\(', '$ $').replace('\(', '$').replace('\)', '$')
        dollar_sequence = dollar_sequence.replace('\\]\\[', '$ $').replace('\\[', '$').replace('\\]', '$')
        # 在$符号前后添加空格跟字母分开，不然会被当成一个普通的$符号
        dollar_sequence = re.sub(r'([a-zA-Z]+)\$', r'\1 $', dollar_sequence)
        dollar_sequence = re.sub(r'\$([a-zA-Z]+)', r'$ \1', dollar_sequence)
        dollar_sequence = dollar_sequence.replace('****', '** **').replace('_-', '-_')
        return dollar_sequence]

    def ocr(self, img_paths=None, images=None, batch_size=8):
        if img_paths:
            images = [Image.open(img_path).convert("RGB") for img_path in img_paths]
        elif not images:
            raise 'You need to input img_path or image'
        else:
            images = [image.convert("RGB") for image in images]
        
        all_outputs = []
        for i in tqdm(range(0, len(images), batch_size), desc='nougat', ncols=50):
            try:
                # 将待处理图片拼接成一个大的tensor
                batch_images = images[i:i+batch_size]
                batch_pixel_values = torch.cat(
                    [self.processor(image, return_tensors="pt").pixel_values for image in batch_images], 
                    0
                )
                
                # generate transcription
                outputs = self.model.generate(
                    batch_pixel_values.to(self.device),
                    min_length=1,
                    max_new_tokens=6000,
                    bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                )
                all_outputs.append(outputs)
                
            except (RuntimeError, AssertionError) as e:
                    
                # 确保存储目录存在
                os.makedirs("./images", exist_ok=True)
                
                # 保存当前batch的图片
                for j, img in enumerate(batch_images):
                    save_path = f"./images/error_batch_{i}_image_{j}.png"
                    img.save(save_path)
                print(f"Saved problematic images to ./images/")
                
                # 记录完整的错误堆栈
                with open("./images/error_log.txt", "a") as f:
                    f.write(f"\nError at batch {i}:\n")
                    f.write(traceback.format_exc())
                
                # 终止执行
                return None
                
        all_sequences = [
            self.processor.batch_decode(outputs, skip_special_tokens=True) for outputs in all_outputs
        ]
        all_sequences = list(chain(*all_sequences))
        dollar_sequences = [self.clean_sequence(sequence) for sequence in all_sequences]
        
        return dollar_sequences

if __name__=='__main__':
    # img_path = '/home/sanxinshidai/liuye/detect/pdf_detect/object_detection/result.png'
    # nougat = NougatAPI()
    # print(nougat.api(img_path))
    nougat_path = '/data1/nzw/model/nougat-base'
    nougat = NougatOCR(nougat_path)
    print(nougat.model)