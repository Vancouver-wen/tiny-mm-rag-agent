import os
import pickle
import json
from typing import List, Any, Tuple

import jieba
from tqdm import tqdm
from joblib import Parallel,delayed

from tinyrag.searcher.bm25_recall.rank_bm25 import OkapiBM25


class BM25Retriever:
    def __init__(self, txt_list: List[str]=[], base_dir="data/db/bm_corpus") -> None:
        self.data_list = txt_list
        self.base_dir = base_dir
        
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir, exist_ok=True)
        
    def build(self, docs: List[str]):
        self.data_list = docs
        self.tokenized_corpus = Parallel(n_jobs=-1,backend="threading")(
            delayed(self.tokenize)(doc)
            for doc in tqdm(docs)
        )
        # for doc in tqdm(self.data_list, desc="bm25 build "):
        #     self.tokenized_corpus.append(self.tokenize(doc))
        # 初始化 BM25Okapi 实例
        self.bm25 = OkapiBM25(self.tokenized_corpus)

    def tokenize(self,  text: str) -> List[str]:
        """ 
        使用jieba进行中文分词。
        """
        result=list(jieba.cut_for_search(text))
        return result

    def save(self, db_name=""):
        """ 
        对数据进行分词并保存到json文件中
        """
        db_name = db_name if db_name != "" else "bm25_data"
        db_file_path = os.path.join(self.base_dir, db_name + ".json")
        # 保存分词结果
        data_to_save = {
            "data_list": self.data_list,
            "tokenized_corpus": self.tokenized_corpus
        }
        
        with open(db_file_path, 'w',encoding='UTF-8') as f:
            json.dump(data_to_save, f,ensure_ascii=False,indent=4)

    def load(self, db_name=""):
        """ 
        从文件中读取分词后的语料库，并重新初始化 BM25Okapi 实例。
        """
        db_name = db_name if db_name != "" else "bm25_data"
        db_file_path = os.path.join(self.base_dir, db_name + ".json")
        
        with open(db_file_path, 'r',encoding="UTF-8") as f:
            data = json.load(f)
        
        self.data_list = data["data_list"]
        self.tokenized_corpus = data["tokenized_corpus"]
        
        # 重新初始化 BM25Okapi 实例
        self.bm25 = OkapiBM25(self.tokenized_corpus)
    
    def search(self, query: str, top_n=5) -> List[Tuple[int, str, float]]:
        """ 
        使用BM25算法检索最相似的文本。
        """
        if self.tokenized_corpus is None:
            raise ValueError("Tokenized corpus is not loaded or generated.")

        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # 获取分数最高的前 N 个文本的索引
        top_n_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]

        # 构建并返回结果列表
        result = [
            (i, self.data_list[i], scores[i])
            for i in top_n_indices
        ]

        return result


