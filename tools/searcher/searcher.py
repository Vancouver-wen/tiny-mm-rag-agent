import os
import json
import copy
from loguru import logger
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from tools.searcher.retriever_bm25 import BM25Retriever
from tools.searcher.retriever_embed import EmbRetriever
from tools.searcher.reanker import RerankerBGEM3

def process_text(doc, emb_model, emb_retriever):
    doc_emb = emb_model.get_embedding(doc)
    emb_retriever.insert(doc_emb, doc)

class Searcher:
    def __init__(self, emb_model_id: str, ranker_model_id: str, device:str="cpu", base_dir: str="data/db") -> None:
        self.base_dir = base_dir
        emb_model_id = emb_model_id
        ranker_model_id = ranker_model_id
        device = device
        
        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)
        
        # 召回
        # bm25召回
        self.bm25_retriever = BM25Retriever(base_dir=self.base_dir+"/bm_corpus")
        # 向量召回
        self.emb_model = HFSTEmbedding(path = emb_model_id,device=device)
        index_dim = len(self.emb_model.get_embedding("test_dim"))
        self.emb_retriever = EmbRetriever(index_dim=index_dim, base_dir=self.base_dir+"/faiss_idx")

        # 排序
        self.ranker = RerankerBGEM3(model_id_key = ranker_model_id, device=device)


    def build_db(self, docs: List[str]):
        # 构建 BM25 索引
        self.bm25_retriever.build(docs)
        logger.info("bm25 retriever build success...")
        # 构建 向量索引
        doc_embs=self.emb_model.get_batch_embedding(docs)
        self.emb_retriever.batch_insert(doc_embs,docs)
        # for doc in tqdm(docs, desc="emb build "):
        #     doc_emb = self.emb_model.get_embedding(doc)
        #     self.emb_retriever.insert(doc_emb, doc)
        logger.info("emb retriever build success...")
        
    def save_db(self):
        # self.base_dir = base_dir
        self.bm25_retriever.save()
        logger.info("bm25 retriever save success...")
        self.emb_retriever.save()
        logger.info("emb retriever save success...")

    def load_db(self):
        # self.base_dir = base_dir
        self.bm25_retriever.load()
        logger.info("bm25 retriever load success...")
        self.emb_retriever.load()
        logger.info("emb retriever load success...")

    def search(self, query:str, top_n=3) -> list:
        # bm25召回 结果
        bm25_recall_list = self.bm25_retriever.search(query, 2 * top_n)
        logger.info("bm25 recall text num: {}".format(len(bm25_recall_list)))
        # for text in bm25_recall_list:
        #     print(text)
        # 向量召回 结果
        query_emb = self.emb_model.get_embedding(query)
        emb_recall_list = self.emb_retriever.search(query_emb, 2 * top_n)
        logger.info("emb recall text num: {}".format(len(emb_recall_list)))
        # for text in emb_recall_list:
        #     print(text)
        # 合并 bm25召回 与 向量召回 的结果 并去重
        recall_unique_text = set()
        for idx, text, score in bm25_recall_list:
            recall_unique_text.add(text)

        for idx, text, score in emb_recall_list:
            recall_unique_text.add(text)

        logger.info("unique recall text num: {}".format(len(recall_unique_text)))

        # 调用 排序模型 对 召回结果 进行排序
        rerank_result = self.ranker.rank(query, list(recall_unique_text), top_n)

        return rerank_result