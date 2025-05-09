import os
import pickle
import json
from typing import List, Any, Tuple

import jieba
from tqdm import tqdm
from joblib import Parallel,delayed


import math
import numpy as np
from multiprocessing import Pool, cpu_count

class BM25:
    """ 每种算法都有其适用场景，选择哪种算法取决于具体的应用需求和数据特性。
        - 如果数据集中存在很多低频词语，那么BM25Okapi可能更适合；
        - 而对于文档长度差异较大的数据集，BM25L或BM25Plus可能表现更好。
    """
    def __init__(self, corpus, tokenizer=None):
        self.corpus_size = 0        # 文档总数
        self.avgdl = 0              # 文档平均长度
        self.doc_freqs = []         # 每个文档中词语的频率
        self.idf = {}               # 逆文档频率（IDF）
        self.doc_len = []           # 每个文档长度
        self.tokenizer = tokenizer

        if tokenizer:
            corpus = self._tokenize_corpus(corpus)
        
        # 初始化文档频率字典
        nd = self._initialize(corpus)
        # 计算逆文档频率（IDF）
        self._calc_idf(nd)

    def _initialize(self, corpus):
        """ 
        初始化文档词频词典
        """
        nd = {}         # 词语 -> 包含该词语的文档数
        num_doc = 0     # 文档总词数
        for document in corpus:
            self.doc_len.append(len(document))
            num_doc += len(document)

            # 计算每个文档中词语的频率
            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            # 添加词语频率到列表中
            self.doc_freqs.append(frequencies)

            # 更新文档频率字典
            for word, freq in frequencies.items():
                try:
                    nd[word] += 1
                except KeyError:
                    nd[word] = 1
            
            self.corpus_size += 1

        # 计算平均文档长度
        self.avgdl = num_doc / self.corpus_size
        return nd
    
    def _tokenize_corpus(self, corpus):
        """ 分词
        """
        pool = Pool(cpu_count())
        tokenized_corpus = pool.map(self.tokenizer, corpus)
        return tokenized_corpus

    def _calc_idf(self, nd):
        """ 计算逆文档频率（IDF）
        """
        raise NotImplementedError()

    def get_scores(self, query):
        """ 计算 query 与文档的相关性得分
        """
        raise NotImplementedError()

    def get_batch_scores(self, query, doc_ids):
        """ 计算 query 与一批文档的相关性得分
        """
        raise NotImplementedError()
    
    def get_top_n(self, query, documents, n=5):
        """ 获取 top-n
        """
        
        assert self.corpus_size == len(documents), "The documents given don't match the index corpus!"
 
        scores = self.get_scores(query)
        # 获取得分最高的n个文档的索引
        top_n = np.argsort(scores)[::-1][:n]
        # 根据索引获取文档
        return [documents[i] for i in top_n]
    
class OkapiBM25(BM25):
    """ 经典的BM25实现 通过设置IDF的下限来处理稀有词语的情况。
    """
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1    # 控制文档频率的影响程度
        self.b = b      # 控制文档长度的影响程度
        self.epsilon = epsilon  # IDF值的下限因子
        super().__init__(corpus, tokenizer)
        

    def _calc_idf(self, nd):
        """ 计算文档和语料库中术语的频率。
            该算法将 idf 值的下限设置为 eps *average_idf
        """
        idf_sum = 0         # 逆文档频率之和
        negative_idfs = []  # 存储IDF值小于0的词语
        for word, freq in nd.items():
            # 计算逆文档频率（IDF）
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)

        # 计算平均逆文档频率
        self.average_idf = idf_sum / len(self.idf)

        # 设置IDF值的下限
        eps = self.epsilon * self.average_idf
        # 将IDF值小于0的词语设置为下限值
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query):
        """ 
        计算 query 与文档的相关性得分。
        score是一个numpy数组 长度为文档数量
        """
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        
        return score
    
    def get_batch_scores(self, query, doc_ids):
        """ 计算查询与指定文档集的相关性得分。
        """
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return score.tolist()

class BowRetrieverBM25:
    def __init__(self, base_dir="data/db/bm_corpus") -> None:
        self.data_list=None
        self.chunks=None
        self.base_dir = base_dir
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir, exist_ok=True)
    
    def chunk2doc(self,chunk:list[dict])->str:
        text=""
        for item in chunk:
            if item['type']=='text':
                text=text+item['text']
        return text
        
    def build(self, chunks:list[list[dict]]):
        self.chunks=chunks
        docs=[self.chunk2doc(chunk) for chunk in chunks]
        self.data_list = docs
        self.tokenized_corpus = Parallel(n_jobs=-1,backend="threading")(
            delayed(self.tokenize)(doc)
            for doc in tqdm(docs)
        )
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
            "chunks":self.chunks,
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
        
        self.chunks = data["chunks"]
        self.data_list = data["data_list"]
        self.tokenized_corpus = data["tokenized_corpus"]
        
        # 重新初始化 BM25Okapi 实例
        self.bm25 = OkapiBM25(self.tokenized_corpus)
    
    def search(self, chunk:list[dict], top_n=5) -> List[Tuple[int,float,list[dict]]]:
        """ 
        使用BM25算法检索最相似的文本。
        """
        if self.tokenized_corpus is None:
            raise ValueError("Tokenized corpus is not loaded or generated.")

        query = self.chunk2doc(chunk)
        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # 获取分数最高的前 N 个文本的索引
        top_n_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
        # self.data_list[i] 与 self.chunks[i] 的文字内容应该相同

        # 构建并返回结果列表
        result = [
            (i,scores[i],self.chunks[i])
            for i in top_n_indices
        ]

        return result


