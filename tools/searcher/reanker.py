import os
import json
import requests

import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoModel

class RerankerBGEM3:
    def __init__(self, model_id_key: str, device: str = "", is_api=False) -> None:
        
        self.device = torch.device(device if device else "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_id_key = model_id_key
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id_key)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id_key)
        self.model.to(self.device)  # 将模型移动到指定设备
        self.model.eval()  # 设置模型为评估模式

    def rank(self, query: str, candidate_query, top_n=3) -> list[tuple[float, str]]:
        # 创建查询和文本对
        pairs = [[query, txt] for txt in candidate_query]

        # 计算得分
        with torch.no_grad():  # 不计算梯度以节省内存
            inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=1024).to(self.device)
            outputs = self.model(**inputs, return_dict=True)
            # outputs.squeeze 就是模型的结果 也就是相关性分数
            scores = outputs.logits.squeeze().cpu().numpy()

        # 将得分和文本对结合，并按得分排序
        scored_query_list = list(zip(scores, candidate_query))
        scored_query_list.sort(key=lambda x: x[0], reverse=True)  # 按得分降序排列

        # 取前 top_n 的结果
        top_n_results = scored_query_list[:top_n]

        return top_n_results


class RerankerJina:
    def __init__(self,model_path:str,device:str):
        self.model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=model_path,
            torch_dtype="auto",
            trust_remote_code=True,
            # attn_implementation="flash_attention_2"
        )

        self.model.to(device)  # or 'cpu' if no GPU is available
        self.model.eval()
        self.example()

    def rank(self,query:str,candidates:list[dict]):
        """
        :query 字符串
        :candidates  chunks块
        chunk=[
            {"type":"text","text":"content"},
            {"type":"image","image":"image path/url"},
            {"type":"text","text":"content"},
            {"type":"image","image":"image path/url"},
        ]
        """
        scores=[]
        for chunk in candidates:
            text=""
            images=[]
            for item in chunk:
                if item['type']=='text':
                    text=text+item['text']
                elif item['type']=='image':
                    images.append(item['image'])
                else:
                    raise NotImplementedError()
            text_score=self.model.compute_score([[query,text]],max_length=2048, doc_type="text")
            if images:
                image_score=self.model.compute_score(
                    [[query,image] for image in images],
                    max_length=2048, doc_type="image"
                ) # .mean()
                if len(images)>1:
                    image_score=float(np.mean(image_score))
                score=(text_score+image_score)/2
            else:
                score=text_score
            scores.append(score)
        return scores
    
    def example(self):
        # text 2 image
        query = "slm markdown"
        documents = [
            "/data/wzh_fd/workspace/tiny-mm-rag-agent/data/tmp_dfcf/2024_Tesla_Cybertruck_Foundation_Series,_front_left_(Greenwich).jpg",
            "/data/wzh_fd/workspace/tiny-mm-rag-agent/data/tmp_dfcf/Tesla_Cybertruck_damaged_window.jpg",
        ]
        image_pairs = [[query, doc] for doc in documents]
        scores = self.model.compute_score(image_pairs, max_length=2048, doc_type="image")
        # [0.49375027418136597, 0.7889736890792847, 0.47813892364501953, 0.5210812091827393]
        
        # text 2 text
        query = "slm markdown"
        documents = [
            "We present ReaderLM-v2, a compact 1.5 billion parameter language model designed for efficient web content extraction. Our model processes documents up to 512K tokens, transforming messy HTML into clean Markdown or JSON formats with high accuracy -- making it an ideal tool for grounding large language models. The models effectiveness results from two key innovations: (1) a three-stage data synthesis pipeline that generates high quality, diverse training data by iteratively drafting, refining, and critiquing web content extraction; and (2) a unified training framework combining continuous pre-training with multi-objective optimization. Intensive evaluation demonstrates that ReaderLM-v2 outperforms GPT-4o-2024-08-06 and other larger models by 15-20% on carefully curated benchmarks, particularly excelling at documents exceeding 100K tokens, while maintaining significantly lower computational requirements.",
            "数据提取么？为什么不用正则啊，你用正则不就全解决了么？",
            "During the California Gold Rush, some merchants made more money selling supplies to miners than the miners made finding gold.",
            "Die wichtigsten Beiträge unserer Arbeit sind zweifach: Erstens führen wir eine neuartige dreistufige Datensynthese-Pipeline namens Draft-Refine-Critique ein, die durch iterative Verfeinerung hochwertige Trainingsdaten generiert; und zweitens schlagen wir eine umfassende Trainingsstrategie vor, die kontinuierliches Vortraining zur Längenerweiterung, überwachtes Feintuning mit spezialisierten Kontrollpunkten, direkte Präferenzoptimierung (DPO) und iteratives Self-Play-Tuning kombiniert. Um die weitere Forschung und Anwendung der strukturierten Inhaltsextraktion zu erleichtern, ist das Modell auf Hugging Face öffentlich verfügbar.",
        ]
        text_pairs = [[query, doc] for doc in documents]
        scores = self.model.compute_score(text_pairs, max_length=1024, doc_type="text")
        
        # image 2 text
        query = "/data/wzh_fd/workspace/tiny-mm-rag-agent/data/tmp_dfcf/paper-11.png"
        documents = [
            "We present ReaderLM-v2, a compact 1.5 billion parameter language model designed for efficient web content extraction. Our model processes documents up to 512K tokens, transforming messy HTML into clean Markdown or JSON formats with high accuracy -- making it an ideal tool for grounding large language models. The models effectiveness results from two key innovations: (1) a three-stage data synthesis pipeline that generates high quality, diverse training data by iteratively drafting, refining, and critiquing web content extraction; and (2) a unified training framework combining continuous pre-training with multi-objective optimization. Intensive evaluation demonstrates that ReaderLM-v2 outperforms GPT-4o-2024-08-06 and other larger models by 15-20% on carefully curated benchmarks, particularly excelling at documents exceeding 100K tokens, while maintaining significantly lower computational requirements.",
            "数据提取么？为什么不用正则啊，你用正则不就全解决了么？",
            "During the California Gold Rush, some merchants made more money selling supplies to miners than the miners made finding gold.",
            "Die wichtigsten Beiträge unserer Arbeit sind zweifach: Erstens führen wir eine neuartige dreistufige Datensynthese-Pipeline namens Draft-Refine-Critique ein, die durch iterative Verfeinerung hochwertige Trainingsdaten generiert; und zweitens schlagen wir eine umfassende Trainingsstrategie vor, die kontinuierliches Vortraining zur Längenerweiterung, überwachtes Feintuning mit spezialisierten Kontrollpunkten, direkte Präferenzoptimierung (DPO) und iteratives Self-Play-Tuning kombiniert. Um die weitere Forschung und Anwendung der strukturierten Inhaltsextraktion zu erleichtern, ist das Modell auf Hugging Face öffentlich verfügbar.",
        ]
        image_pairs = [[query, doc] for doc in documents]
        scores = self.model.compute_score(image_pairs, max_length=2048, query_type="image", doc_type="text")  
        
        # image 2 image
        query = "/data/wzh_fd/workspace/tiny-mm-rag-agent/data/tmp_dfcf/paper-11.png"
        documents = [
            "/data/wzh_fd/workspace/tiny-mm-rag-agent/data/tmp_dfcf/Tesla_Cybertruck_damaged_window.jpg",
            "/data/wzh_fd/workspace/tiny-mm-rag-agent/data/tmp_dfcf/paper-11.png",
            "/data/wzh_fd/workspace/tiny-mm-rag-agent/data/tmp_dfcf/2024_Tesla_Cybertruck_Foundation_Series,_front_left_(Greenwich).jpg",
        ]
        image_pairs = [[query, doc] for doc in documents]
        scores = self.model.compute_score(image_pairs, max_length=2048, doc_type="image", query_type='image')
        

class RemoteRerankerJina:
    def __init__(self,url:str):
        self.url=url
        self.example()
        
    def rank(self,query:str,candidates:list[dict]):
        input_data = {
            "data": [query,candidates]
        }
        scores = json.loads(requests.post(self.url, json=input_data).json())
        return scores
    
    def example(self):
        query = "/data/wzh_fd/workspace/tiny-mm-rag-agent/data/tmp_dfcf/paper-11.png"
        chunks = [
            [{'type':'text','text':"We present ReaderLM-v2, a compact 1.5 billion parameter language model designed for efficient web content extraction. Our model processes documents up to 512K tokens, transforming messy HTML into clean Markdown or JSON formats with high accuracy -- making it an ideal tool for grounding large language models. The models effectiveness results from two key innovations: (1) a three-stage data synthesis pipeline that generates high quality, diverse training data by iteratively drafting, refining, and critiquing web content extraction; and (2) a unified training framework combining continuous pre-training with multi-objective optimization. Intensive evaluation demonstrates that ReaderLM-v2 outperforms GPT-4o-2024-08-06 and other larger models by 15-20% on carefully curated benchmarks, particularly excelling at documents exceeding 100K tokens, while maintaining significantly lower computational requirements."}],
            [{'type':'text','text':"数据提取么？为什么不用正则啊，你用正则不就全解决了么？"}],
            [{'type':'text','text':"During the California Gold Rush, some merchants made more money selling supplies to miners than the miners made finding gold."}],
            [{'type':'text','text':"Die wichtigsten Beiträge unserer Arbeit sind zweifach: Erstens führen wir eine neuartige dreistufige Datensynthese-Pipeline namens Draft-Refine-Critique ein, die durch iterative Verfeinerung hochwertige Trainingsdaten generiert; und zweitens schlagen wir eine umfassende Trainingsstrategie vor, die kontinuierliches Vortraining zur Längenerweiterung, überwachtes Feintuning mit spezialisierten Kontrollpunkten, direkte Präferenzoptimierung (DPO) und iteratives Self-Play-Tuning kombiniert. Um die weitere Forschung und Anwendung der strukturierten Inhaltsextraktion zu erleichtern, ist das Modell auf Hugging Face öffentlich verfügbar."}],
        ]
        input_data = {
            "data": [query,chunks]
        }
        scores = json.loads(requests.post(self.url, json=input_data).json())
        scores = np.array(scores,dtype=np.float32)