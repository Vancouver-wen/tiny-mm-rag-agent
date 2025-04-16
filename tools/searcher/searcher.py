import os
import json
import copy

from loguru import logger
from tqdm import tqdm
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from natsort import natsorted

from tools.searcher.retriever_bm25 import BowRetrieverBM25
from tools.searcher.retriever_embed import EmbEncoderGme,EmbRetrieverFaiss
from tools.searcher.reanker import RerankerJina


class Searcher:
    def __init__(
        self, 
        base_dir: str, # 保存召回索引的目录 ="data/db"
        embedding_model:str,
        reranker_model:str,
        device:str
    ) -> None:
        self.base_dir = base_dir
        
        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)
        
        # 召回
        # bm25召回
        self.bm25_retriever = BowRetrieverBM25(base_dir=self.base_dir+"/bm_corpus")
        # 向量召回
        self.emb_model = EmbEncoderGme(embedding_model,device=device)
        index_dim = self.emb_model.hidden_size
        self.emb_retriever = EmbRetrieverFaiss(index_dim=index_dim, base_dir=self.base_dir+"/faiss_idx")

        # 排序
        self.ranker = RerankerJina(reranker_model,device=device)


    def build_db(self, chunks: list[list[dict]]):
        # 构建 向量索引
        logger.info("emb retriever building. it may take a long time ...")
        for chunk in tqdm(chunks):
            embedding=self.emb_model.encode(chunk)
            embedding=embedding.tolist()
            self.emb_retriever.insert(embedding,chunk)
        # 构建 BM25 索引
        self.bm25_retriever.build(chunks)
        
    def save_db(self):
        self.bm25_retriever.save()
        self.emb_retriever.save()

    def load_db(self):
        self.bm25_retriever.load()
        self.emb_retriever.load()

    def search(self, chunk:list[dict], top_n=3, query="") -> list:
        # bm25召回 结果
        bm25_recall_list = self.bm25_retriever.search(chunk, 2 * top_n)
        # for text in bm25_recall_list:
        #     print(text)
        # 向量召回 结果
        query_emb = self.emb_model.encode(chunk)
        emb_recall_list = self.emb_retriever.search(query_emb, 2 * top_n)
        # for text in emb_recall_list:
        #     print(text)
        
        # 合并 bm25召回 与 向量召回 的结果 并去重
        recall_unique_index = set()
        recall_unique_chunk = list()
        for idx, score, chunk in bm25_recall_list+emb_recall_list:
            if not recall_unique_index.__contains__(idx):
                recall_unique_index.add(idx)
                recall_unique_chunk.append(chunk)

        # 调用 排序模型 对 召回结果 进行排序
        if not query:
            for item in chunk:
                if item['type']=='text':
                    query=query+'\n'+item['text']
        scores = self.ranker.rank(query, recall_unique_chunk)

        return sorted(list(zip(scores,recall_unique_chunk)),key=lambda x:-x[0])[:top_n]