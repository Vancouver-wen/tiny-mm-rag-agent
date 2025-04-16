import os
os.environ['CUDA_VISIBLE_DEVICES']='2,3'
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Disabling parallelism to avoid deadlocks...
import argparse
import json

import torch
import yaml
import jsonlines
from easydict import EasyDict
from loguru import logger
from tqdm import tqdm
from joblib import Parallel,delayed
from tools.llm.qwenvl import QwenVL
from tools.searcher.searcher import Searcher

class TinyRAG:
    def __init__(self, config) -> None:
        logger.info(f"config: {config}")
        self.config = config
        self.searcher = Searcher(
            base_dir=os.path.dirname(self.config.chunk_file),
            embedding_model=self.config.embedding_model,
            reranker_model=self.config.reranker_model,
            device="cuda:1"
        )
        self.llm = QwenVL(
            model_path=self.config.llm_model,
            device="cuda:0"
        )

    def build(self, chunks:list[dict]):
        """ 构建数据库可能需要很长时间"""
        self.searcher.build_db(chunks)
        self.searcher.save_db()

    def load(self):
        self.searcher.load_db()

    def chat(self) -> str:
        """
        一开始不要想着自动化，先手动控制流程
        """
        messages = [
            {
                "role": "user",
                "content": [
                    # {"type": "image","image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",},
                    {"type": "text", "text": "今天是2025年5月1日, 我想知道今天的汽车行业投资建议。"},
                ],
            }
        ]
        # LLM的初次回答
        response = self.llm.generate(messages)
        messages.append({
            'role':'assistent',
            'content':[{'type':'text','text':response}]
        })
        # 数据库检索的文本
        ## 拼接 query和LLM初次生成的结果，查找向量数据库
        chunk=messages[0]['content']+messages[1]['content']
        search_content_list = self.searcher.search(chunk, top_n=6) # list of (score,chunk)
        # 构造 输入
        messages.append({
            'role':"user",
            'content':[{'type':'text','text':'--- \n参考信息: \n'}]
        })
        for score,chunk in search_content_list:
            messages[-1]['content']=messages[-1]['content']+chunk
        messages[-1]['content'].append({
            'type':'text',
            'text':"""
---
请根据上述参考信息回答，修正之前的回答。
前面的参考信息和回答可能有用，也可能没用，你需要从我给出的参考信息中选出与我的问题最相关的那些，来为你修正的回答提供依据。
你修正的回答:
"""
        })
        # 生成最终答案
        response = self.llm.generate(messages)
        messages.append({
            'role':'assistent',
            'content':[{'type':'text','text':response}]
        })
        return messages


def main(args):
    with open(args.config,'r', encoding='utf-8') as f:
        config=EasyDict(yaml.safe_load(f))
    tiny_rag = TinyRAG(config)

    if args.build:
        chunks=[]
        chunk_file=config.chunk_file
        with jsonlines.open(chunk_file,'r') as reader:
            for obj in reader:
                chunks.append(obj)
        # chunks=chunks[:100] # fast debug
        tiny_rag.build(chunks)
    else:
        tiny_rag.load()
    
    # 这里可以测试 llm 的任务拆解能力，以及多步执行能力
    messages=tiny_rag.chat() 
    with open("./temp.json",'w',encoding="UTF-8") as f:
        json.dump(messages,f,ensure_ascii=False,indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tiny RAG Argument Parser')
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--config', type=str, default="config/config.yaml", help='Tiny RAG config')

    args = parser.parse_args()
    
    main(args)

