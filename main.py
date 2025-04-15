import os
import argparse

import yaml
import jsonlines
from loguru import logger
from tqdm import tqdm
from joblib import Parallel,delayed

from tools.llm.deepseek import DeepSeek
from tools.searcher.searcher import Searcher

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


class TinyRAG:
    def __init__(self, config) -> None:
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
        self.llm = DeepSeek(
            model_id_key=config.llm_model_id,
            device=self.config.device,
        )

    def build(self, chunks:list[dict]):
        """ 构建数据库可能需要很长时间"""
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
        

    

def main(args):
    with open(args.config,'r', encoding='utf-8') as f:
        config=yaml.safe_load(f)
    tiny_rag = TinyRAG(config)

    if args.build:
        chunks=[]
        chunk_path=args.chunk_path
        with jsonlines.open(chunk_path,'r') as reader:
            for obj in reader:
                chunks.append(obj)
        tiny_rag.build(chunks)
    else:
        tiny_rag.load()
    
    query = "首先介绍一下你是谁？ 然后介绍农业银行。"
    top_n=6
    output = tiny_rag.search(query, top_n)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tiny RAG Argument Parser')
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--config', type=str, default="config/config.yaml", help='Tiny RAG config')

    args = parser.parse_args()
    
    main(args)

