import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Disabling parallelism to avoid deadlocks...
import argparse
import json

import yaml
from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
from easydict import EasyDict

from tools.searcher.retriever_embed import EmbEncoderGme
from tools.searcher.reanker import RerankerJina
from tools.llm.qwenvl import QwenVL

app = FastAPI()

# 定义请求体的模型
class InputData(BaseModel):
    data: list

embedder = None
ranker = None
llm = None

@app.post("/embedding")
async def embedder_process(input_data: InputData):
    chunk = input_data.data
    # print(chunk)
    try:
        query_emb = embedder.encode(chunk)
        # print(query_emb)
        query_emb = json.dumps(query_emb,ensure_ascii=False)
    except Exception as e:
        query_emb=str(e)
    return query_emb

@app.post("/reranker")
async def ranker_process(input_data: InputData):
    query,recall_unique_chunk = input_data.data
    # print(query)
    # print(recall_unique_chunk)
    try:
        scores=ranker.rank(query, recall_unique_chunk)
        scores = json.dumps(scores,ensure_ascii=False)
        # print(scores)
    except Exception as e:
        scores=str(e)
    return scores

@app.post("/llm")
async def llm_process(input_data: InputData):
    messages = input_data.data
    # print(messages)
    try:
        response = llm.generate(messages)
    except Exception as e:
        response=str(e)
    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Server Argument Parser')
    parser.add_argument('--config', type=str, default="config/config.yaml", help='Tiny RAG config')

    args = parser.parse_args()
    
    config_path=args.config
    
    with open(config_path,'r',encoding="UTF-8") as f:
        config=yaml.safe_load(f)
    config=EasyDict(config)
    
    if not embedder:
        embedder = EmbEncoderGme(config.embedding_model,device="cuda:0") # "cuda:0"
        
    if not ranker:
        ranker = RerankerJina(config.reranker_model,device="cuda:0") # "cuda:0"
    
    if not llm:
        llm = QwenVL(
            model_path=config.llm_model,
            device="balanced_low_0" # "balanced_low_0"
        )
        
    uvicorn.run(app, host="0.0.0.0", port=4514)
    
    
    
