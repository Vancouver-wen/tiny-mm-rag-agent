import os
import sys
import json
import random
import argparse
sys.path.append(".")

from loguru import logger
from tqdm import tqdm
from easydict import EasyDict

from tinyrag import RAGConfig, TinyRAG
from tinyrag.utils import read_json,read_yaml
    

def main():
    parser = argparse.ArgumentParser(description='Tiny RAG Argument Parser')
    parser.add_argument("-t", '--type', type=str, default="search", help='Tiny RAG Type [build, search]')
    parser.add_argument("-c", '--config', type=str, default="config/qwen2_config.json", help='Tiny RAG config')
    parser.add_argument('-p', "--path",  type=str, default="data/raw_data/wikipedia-cn-20230720-filtered.json", help='Tiny RAG data path')

    args = parser.parse_args()
    
    config = read_yaml(args.config)
    rag_config = RAGConfig(**config)
    tiny_rag = TinyRAG(config=rag_config)
    logger.info("tiny rag init success!")

    if args.type == "build":
        raw_data_list = read_json(args.path)
        logger.info("load raw data success! ")
        # 数据太多了，随机采样 100 条数据
        # raw_data_part = random.sample(raw_data_list, 100)

        text_list = [item["completion"] for item in raw_data_list]

        tiny_rag.build(text_list)
    elif args.type == "search":
        tiny_rag.load()
        query = "首先介绍一下你是谁？ 然后介绍农业银行。"
        top_n=6
        output = tiny_rag.search(query, top_n)
    else:
        raise NotImplementedError(f"unknown args.type:{args.type}")

if __name__ == "__main__":
    main()

