import os
import json

import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from modelscope import AutoModelForCausalLM,AutoTokenizer,AutoConfig # 从modelscope也会自动链接到transformers包
from loguru import logger


class DeepSeek:
    def __init__(self, model_id_key: str, device: str = "cpu", is_api=False) -> None:
        super().__init__(model_id_key, device, is_api)

        # 从预训练模型加载因果语言模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id_key,  # 模型标识符
            torch_dtype="auto",  # 自动选择张量类型
            device_map=self.device,  # 分布到特定设备上
            trust_remote_code=True  # 允许加载远程代码
        )
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id_key,  # 分词器标识符
            trust_remote_code=True,
        )
        # import pdb;pdb.set_trace()
        # self.tokenizer.get_added_vocab()
        # self.tokenizer.all_special_tokens
        # 加载配置文件
        self.config = AutoConfig.from_pretrained(
            self.model_id_key,  # 配置文件标识符
            trust_remote_code=True  # 允许加载远程代码
        )

        if self.device == "cpu":
            self.model.float()
        
        # 设置模型为评估模式
        self.model.eval()

    def generate(self, messages:list[dict]) -> str:
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        batch_input_ids=model_inputs['input_ids']
        attention_mask=model_inputs['attention_mask']

        generated_ids = self.model.generate( # 这里封装的太好了 输入是 vocab的index 输出也是 vocab的index
            batch_input_ids,
            max_new_tokens=2048 if self.model.device=="cuda" else 512
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(batch_input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response



