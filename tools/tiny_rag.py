import os
import json
import random
from typing import Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
from dataclasses import dataclass

import jsonlines
from loguru import logger
from tqdm import tqdm

from joblib import Parallel,delayed

from tools import BaseLLM, Qwen2LLM, TinyLLM, DeekSeek
from tools import Searcher
from tools import SentenceSplitter
from tools.utils import write_list_to_jsonl

# 在python中定义字符串可以用 ' ' 或者 " " 或者  """ """
# 区别在于: ' ' 或者 " " 是单行字符串，如果需要换行则需要手动 \n
# """ """ 是多行字符串，因此非常适合用来定义 prompt template
RAG_PROMPT_TEMPLATE="""
参考信息：
{context}
---
我的问题或指令：
{question}
---
我的回答：
{answer}
---
请根据上述参考信息回答和我的问题或指令，修正我的回答。
前面的参考信息和我的回答可能有用，也可能没用，你需要从我给出的参考信息中选出与我的问题最相关的那些，来为你修正的回答提供依据。
回答一定要忠于原文，简洁但不丢信息，不要胡乱编造。我的问题或指令是什么语种，你就用什么语种回复。
你修正的回答:
"""

@dataclass
class RAGConfig:
    base_dir:str = "data/wiki_db"
    llm_model_id:str = "models/tiny_llm_sft_92m"
    emb_model_id: str = "models/bge-small-zh-v1.5"
    ranker_model_id:str = "models/bge-reranker-base"
    device:str = "cpu"
    sent_split_model_id:str = "models/nlp_bert_document-segmentation_chinese-base"
    sent_split_use_model:bool = False
    sentence_size:int = 256
    model_type: str = "tinyllm"

def process_docs_text(docs_text, sent_split_model):
    sent_res = sent_split_model.split_text(docs_text)
    return sent_res

class TinyRAG:
    def __init__(self, config:RAGConfig) -> None:
        print("config: ", config)
        self.config = config
        logger.info(f"=> initing Searcher: emb_model and ranker model ..")
        self.searcher = Searcher(
            emb_model_id=config.emb_model_id,
            ranker_model_id=config.ranker_model_id,
            device=config.device,
            base_dir=config.base_dir
        )
        logger.info(f"=> initing Chatter: large language model ..")
        self.llm:BaseLLM = DeekSeek(
            model_id_key=config.llm_model_id,
            device=self.config.device,
        )

    def build(self, chunks:list[dict]):
        """ 
        注意： 构建数据库需要很长时间
        """
        self.searcher.build_db(chunks)
        self.searcher.save_db()

    def load(self):
        self.searcher.load_db()

    def search(self, query: str, top_n:int = 3) -> str:
        # LLM的初次回答
        llm_result_txt = self.llm.generate(query)
        # 数据库检索的文本
        ## 拼接 query和LLM初次生成的结果，查找向量数据库
        search_content_list = self.searcher.search(query=query+llm_result_txt+query, top_n=top_n)
        content_list = [item[1] for item in search_content_list]
        context = "\n".join(content_list)
        # 构造 prompt
        prompt_text = RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=query,
            answer=llm_result_txt
        )
        # 生成最终答案
        output = self.llm.generate(prompt_text)
        logger.info(f"=> Query: {query}")
        logger.info(f"=> Retrieve Top - {top_n}")
        logger.info(f"=> OriginOutput: {llm_result_txt}")
        logger.info(f"=> Prompt: {prompt_text}")
        logger.info(f"=> Output: {output}")
        return output
        


