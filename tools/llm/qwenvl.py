import os
import requests

import torch
from transformers import AutoModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from loguru import logger

class QwenVL:
    def __init__(self,model_path:str,device:str,min_pixels = 25*28*28,max_pixels = 64*28*28):
        # default: Load the model on the available device(s)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=model_path, 
            torch_dtype="auto", 
            device_map=device,
        )
        self.model.eval()

        # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
        # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        #     "Qwen/Qwen2.5-VL-7B-Instruct",
        #     torch_dtype=torch.bfloat16,
        #     attn_implementation="flash_attention_2",
        #     device_map="auto",
        # )
        
        # default processor
        self.processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path=model_path, 
            min_pixels=min_pixels, 
            max_pixels=max_pixels,
        )

        # The default range for the number of visual tokens per image in the model is 4-16384.
        # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
        # min_pixels = 256*28*28
        # max_pixels = 1280*28*28
        # processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
        self.example()

    def generate(self,messages:list[dict],max_new_tokens=2048)->str:
        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = output_text[0]
        return response

    def example(self,):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {"type": "image", "image": "/data/wzh_fd/workspace/tiny-mm-rag-agent/data/tmp_dfcf/Tesla_Cybertruck_damaged_window.jpg"}
                ],
            }
        ]
        logger.info(self.generate(messages))
        
class RemoteQwenVL:
    def __init__(self,url:str):
        self.url=url
        self.example()
        
    def generate(self,messages:list[dict])->str:
        input_data = {
            "data": messages
        }
        response = requests.post(self.url, json=input_data).json()
        return response
        
    def example(self,):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {"type": "image", "image": "/data/wzh_fd/workspace/tiny-mm-rag-agent/data/tmp_dfcf/Tesla_Cybertruck_damaged_window.jpg"}
                ],
            }
        ]
        # 将数据封装成符合服务器要求的格式
        input_data = {
            "data": messages
        }

        # 发送 POST 请求
        response = requests.post(self.url, json=input_data)

        # 打印返回的 JSON 数据
        # print("Response from server:")
        # print(response.json())