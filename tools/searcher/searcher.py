import os
import json
import copy
from loguru import logger
from tqdm import tqdm
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from tools.searcher.retriever_bm25 import BowRetrieverBM25
from tools.searcher.retriever_embed import EmbEncoderGme,EmbRetrieverFaiss
from tools.searcher.reanker import RerankerJina


class Searcher:
    def __init__(
        self, 
        base_dir: str, # 保存召回索引的目录 ="data/db"
        embedding_model:str,
        reranker_model:str,
    ) -> None:
        self.base_dir = base_dir
        
        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)
        
        # 召回
        # bm25召回
        self.bm25_retriever = BowRetrieverBM25(base_dir=self.base_dir+"/bm_corpus")
        # 向量召回
        self.emb_model = EmbEncoderGme(embedding_model)
        index_dim = self.emb_model.hidden_size
        self.emb_retriever = EmbRetrieverFaiss(index_dim=index_dim, base_dir=self.base_dir+"/faiss_idx")

        # 排序
        self.ranker = RerankerJina(reranker_model)


    def build_db(self, chunks: list[list[dict]]):
        # 构建 BM25 索引
        self.bm25_retriever.build(chunks)
        # 构建 向量索引
        logger.info("emb retriever building. it may take a long time ...")
        for chunk in tqdm(chunks):
            doc_emb=self.emb_model.encode(chunk)
            self.emb_retriever.insert(doc_emb,chunk)
        
    def save_db(self):
        self.bm25_retriever.save()
        self.emb_retriever.save()

    def load_db(self):
        self.bm25_retriever.load()
        self.emb_retriever.load()

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