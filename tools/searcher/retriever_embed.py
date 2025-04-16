import os
import json

import faiss
import numpy as np
import torch
import jsonlines

from .gme_inference import GmeQwen2VL

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
        """
        return scores,indexs
        检索距离矩阵D 维度为[nq查询向量样本数,k相似向量数]。其中列表示与k个相似向量的距离 距离从近到远排序。
        检索结果距离I 维度为[nq查询向量样本数,k相似向量数]。其中列表示索引库中k个相似向量的id号 相似度从高到低排序
        """
        vec = np.array(vec, dtype=np.float32)  # 转换为 NumPy 数组
        if vec.ndim == 1:
            vec = np.expand_dims(vec, axis=0)  # 转换为 (1, d) 形状
        D,I=self.index.search(vec, num)
        D=np.squeeze(D)
        I=np.squeeze(I)
        return D.tolist(),I.tolist()

class EmbRetrieverFaiss:
    def __init__(self, index_dim: int, base_dir="data/db/faiss_idx") -> None:
        self.index_dim = index_dim
        self.invert_index = EmbIndex(index_dim)
        self.forward_index = []
        self.base_dir = base_dir
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir, exist_ok=True)

    def insert(self, emb: list, chunk: list[dict]):
        self.invert_index.insert(emb)
        self.forward_index.append(chunk)

    def save(self):
        self.index_name = "index_" + str(self.index_dim)
        self.index_folder_path = os.path.join(self.base_dir, self.index_name)
        if not os.path.exists(self.index_folder_path):
            os.makedirs(self.index_folder_path, exist_ok=True)

        # 保存 faiss 与 index对应的chunk
        self.invert_index.save(self.index_folder_path + "/invert_index.faiss")
        with jsonlines.open(self.index_folder_path + "/forward_index.json", "w") as writer:
            for chunk in self.forward_index:
                writer.write(chunk)
    
    def load(self):
        self.index_name = "index_" + str(self.index_dim)
        self.index_folder_path = os.path.join(self.base_dir, self.index_name)

        self.invert_index = EmbIndex(self.index_dim)
        self.invert_index.load(self.index_folder_path + "/invert_index.faiss")

        self.forward_index = []
        with jsonlines.open(self.index_folder_path + "/forward_index.json",'r') as reader:
            for chunk in reader:
                self.forward_index.append(chunk)

    def search(self, embs: list, top_n=5)->list[tuple[int,float,list[dict]]]:
        """
        return list of [index,score,chunk]
        """
        scores,indexs = self.invert_index.search(embs, top_n)
        recall_list = []
        for i in range(top_n):
            recall_list.append((indexs[i],scores[i], self.forward_index[indexs[i]]))
        return recall_list






class EmbEncoderGme:
    def __init__(self,model_path:str,device:str):
        self.gme = GmeQwen2VL(
            model_path,
            device=device,
            min_image_tokens=128,
            max_image_tokens=256
        )
        self.hidden_size=self.gme.base.config.hidden_size
    
    def encode(self,chunk:list[dict])->torch.Tensor:
        """
        chunk=[
            {"type":"text","text":"content"},
            {"type":"image","image":"image path/url"},
            {"type":"text","text":"content"},
            {"type":"image","image":"image path/url"},
        ]
        """
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

    def example(self):
        texts = [
            "What kind of car is this?",
            "The Tesla Cybertruck is a battery electric pickup truck built by Tesla, Inc. since 2023."
        ]
        images = [
            '/data/wzh_fd/workspace/tiny-mm-rag-agent/data/tmp_dfcf/Tesla_Cybertruck_damaged_window.jpg',
            '/data/wzh_fd/workspace/tiny-mm-rag-agent/data/tmp_dfcf/2024_Tesla_Cybertruck_Foundation_Series,_front_left_(Greenwich).jpg',
        ]
        # Single-modal embedding
        e_text = self.gme.get_text_embeddings(texts=texts)
        e_image = self.gme.get_image_embeddings(images=images)
        print((e_text * e_image).sum(-1))
        ## tensor([0.2281, 0.6001], dtype=torch.float16)

        # How to set embedding instruction
        e_query = self.gme.get_text_embeddings(texts=texts, instruction='Find an image that matches the given text.')
        # If is_query=False, we always use the default instruction.
        e_corpus = self.gme.get_image_embeddings(images=images, is_query=False)
        print((e_query * e_corpus).sum(-1))
        ## tensor([0.2433, 0.7051], dtype=torch.float16)

        # Fused-modal embedding
        e_fused = self.gme.get_fused_embeddings(texts=texts, images=images)
        print((e_fused[0] * e_fused[1]).sum())
        ## tensor(0.6108, dtype=torch.float16)
        import pdb;pdb.set_trace()
        
        
if __name__=="__main__":
    emb_emcoder=EmbEncoderGme("/data/wzh_fd/workspace/Models/gme-Qwen2-VL-2B-Instruct")
    emb_emcoder.example()
    import pdb;pdb.set_trace()