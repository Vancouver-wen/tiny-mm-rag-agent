import os
import json

import faiss
import numpy as np
import torch

from gme_inference import GmeQwen2VL

class EmbEncoder:
    def __init__(self):
        self.gme = GmeQwen2VL("/data/wzh_fd/workspace/Models/gme-Qwen2-VL-2B-Instruct")
    
    def encode(self,chunk:list[dict])->torch.Tensor:
        text=""
        images=[]
        for item in chunk:
            if item['type']=='text':
                text=text+item['text']
            elif item['type']=='image':
                images.append(item['image'])
            else:
                raise NotImplementedError()
        text_embedding=self.gme.get_text_embeddings(
            texts=[text], 
            instruction='Find an image that matches the given text.',
        )
        text_embedding=text_embedding.squeeze()
        if images:
            # If is_query=False, we always use the default instruction.
            image_embeddings=self.gme.get_image_embeddings(
                images=images,
                is_query=False,
                disable_tqdm=False
            )
            image_embedding=image_embeddings.mean(dim=0)
            embedding=torch.mean(torch.stack([text_embedding,image_embedding]),dim=0)
        else:
            embedding=text_embedding
        return embedding

class EmbIndex:
    def __init__(self, index_dim: int) -> None:
        """
        精确检索 IndexFlatL2 使用暴力搜索 brute-force search 的方式
        计算查询向量与索引中所有向量之间的欧几里得距离，返回距离最小的前 k 个向量
        """
        description = "HNSW64"
        measure = faiss.METRIC_L2
        # self.index = faiss.index_factory(index_dim, description, measure)
        self.index = faiss.IndexFlatL2(index_dim)
    def insert(self, emb: list):
        emb = np.array(emb, dtype=np.float32)  # 转换为 NumPy 数组
        if emb.ndim == 1:
            emb = np.expand_dims(emb, axis=0)  # 转换为 (1, d) 形状
        # print("Inserting emb: ", emb)
        # print("Inserting emb: ", emb.shape)
        self.index.add(emb)
        # print("Insertion successful")
    
    def batch_insert(self, embs: list):
        embs = np.array(embs, dtype=np.float32)  # 转换为 NumPy 数组
        if embs.ndim == 1:
            embs = np.expand_dims(embs, axis=0)  # 转换为 (1, d) 形状
        elif embs.ndim == 2 and embs.shape[0] == 1:
            embs = np.squeeze(embs, axis=0)  # 处理 (1, d) 形状
        print("Batch inserting embs: ", embs)
        self.index.add(embs)
        print("Batch insertion successful")

    def load(self, path: str):
        self.index = faiss.read_index(path)

    def save(self, path: str):
        faiss.write_index(self.index, path)

    def search(self, vec: list, num: int):
        vec = np.array(vec, dtype=np.float32)  # 转换为 NumPy 数组
        if vec.ndim == 1:
            vec = np.expand_dims(vec, axis=0)  # 转换为 (1, d) 形状
        return self.index.search(vec, num)

class EmbRetriever:
    def __init__(self, index_dim: int, base_dir="data/db/faiss_idx") -> None:
        self.index_dim = index_dim
        self.invert_index = EmbIndex(index_dim)
        self.forward_index = []
        self.base_dir = base_dir
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir, exist_ok=True)

    def insert(self, emb: list, doc: str):
        self.invert_index.insert(emb)
        self.forward_index.append(doc)

    def save(self, index_name=""):
        self.index_name = index_name if index_name  else "index_" + str(self.index_dim)
        self.index_folder_path = os.path.join(self.base_dir, self.index_name)
        if not os.path.exists(self.index_folder_path):
            os.makedirs(self.index_folder_path, exist_ok=True)

        with open(self.index_folder_path + "/forward_index.txt", "w", encoding="utf8") as f:
            for data in self.forward_index:
                f.write("{}\n".format(json.dumps(data, ensure_ascii=False)))

        self.invert_index.save(self.index_folder_path + "/invert_index.faiss")
    
    def load(self, index_name=""):
        self.index_name = index_name if index_name != "" else "index_" + str(self.index_dim)
        self.index_folder_path = os.path.join(self.base_dir, self.index_name)

        self.invert_index = EmbIndex(self.index_dim)
        self.invert_index.load(self.index_folder_path + "/invert_index.faiss")

        self.forward_index = []
        with open(self.index_folder_path + "/forward_index.txt", encoding="utf8") as f:
            for line in f:
                self.forward_index.append(json.loads(line.strip()))

    def search(self, embs: list, top_n=5):
        search_res = self.invert_index.search(embs, top_n)
        recall_list = []
        for idx in range(top_n):
            recall_list.append((search_res[1][0][idx], self.forward_index[search_res[1][0][idx]], search_res[0][0][idx]))
        return recall_list
