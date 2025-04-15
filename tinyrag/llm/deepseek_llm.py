import os
import json
from typing import Dict, List, Optional, Tuple, Union

import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from modelscope import AutoModelForCausalLM,AutoTokenizer,AutoConfig # 从modelscope也会自动链接到transformers包
from loguru import logger

from tinyrag.llm.base_llm import BaseLLM

class DeekSeekLLM(BaseLLM):
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

    def generate(self, content: str) -> str:
        # 这里的message就是 transformers库的模板 
        # https://huggingface.co/docs/transformers/main/zh/chat_templating
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        logger.info(f"\n => content:{content} \n => apply chat template:{text}")
        """ 每个模型都有自己的  chat template 主要是一些 特殊的token使用 比如 句子开始  人物角色  以及<think>等思维链
        content:你是谁？ 请介绍一下农业银行。 
        apply chat template:<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>你是谁？ 请介绍一下农业银行。<｜Assistant｜><think>
        """

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



