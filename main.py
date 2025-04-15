import os
import sys
import json
import random
import argparse
sys.path.append(".")

from loguru import logger
from tqdm import tqdm
from easydict import EasyDict
import jsonlines

from tools import RAGConfig, TinyRAG
from tools.utils import read_json,read_yaml
    

def main(args):
    config = read_yaml(args.config)
    rag_config = RAGConfig(**config)
    tiny_rag = TinyRAG(config=rag_config)
    logger.info("tiny rag init success!")

    if args.type == "build":
        chunks=[]
        chunk_path=args.chunk_path
        with jsonlines.open(chunk_path,'r') as reader:
            for obj in reader:
                chunks.append(obj)
        tiny_rag.build(chunks)
    elif args.type == "search":
        tiny_rag.load()
        query = "首先介绍一下你是谁？ 然后介绍农业银行。"
        top_n=6
        output = tiny_rag.search(query, top_n)
    else:
        raise NotImplementedError(f"unknown args.type:{args.type}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tiny RAG Argument Parser')
    parser.add_argument("-t", '--type', type=str, default="search", help='Tiny RAG Type [build, search]')
    parser.add_argument("-c", '--config', type=str, default="config/deepseek_config.yaml", help='Tiny RAG config')
    parser.add_argument('-p', "--path",  type=str, default="data/raw_data/wikipedia-cn-20230720-filtered.json", help='Tiny RAG data path')

    args = parser.parse_args()
    
    main(args)

