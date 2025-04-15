from typing import Dict, List, Optional, Tuple, Union

import torch
from sentence_transformers import SentenceTransformer, util
# SentenceTransformers is a Python framework for state-of-the-art sentence, text and image embeddings.
from tqdm import tqdm

from tinyrag.embedding.base_emb import BaseEmbedding


class HFSTEmbedding(BaseEmbedding):
    """
    class for Hugging face sentence embeddings
    """
    def __init__(self, path: str,device:str="cpu", is_api: bool = False) -> None:
        super().__init__(path, is_api)
        self.st_model = SentenceTransformer(path,device=device) # 这里全是调包 输入是huggingFace下载下来的模型文件夹 输出就直接是模型了
        self.name = "hf_model"

    def get_embedding(self, text: str) -> List[float]:
        st_embedding = self.st_model.encode([text], normalize_embeddings=True)
        return st_embedding[0].tolist()
    
    def get_batch_embedding(self,texts:list[str],batch_size=64):
        all_embeddings=[]
        for i in tqdm(range(0, len(texts), batch_size)):
            sub_texts=texts[i:i+batch_size]
            st_embeddings = self.st_model.encode(sub_texts, normalize_embeddings=True)
            for st_embedding in st_embeddings:
                all_embeddings.append(st_embedding.tolist())
        return all_embeddings
