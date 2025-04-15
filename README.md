# tiny-mm-rag-agent
一个很小的多模态rag智能体实现。

项目背景：根据最新的行业调研报告，针对不同的用户，给出个性化的投资建议。

RAG实现：
1. 爬取[东方财富网](https://data.eastmoney.com/report/)的最新行研报告
2. 通过[MinerU](https://github.com/opendatalab/MinerU)将pdf报告解析为markdown文件
3. 使用[langchain-text-splitters](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/markdown_header_metadata/)将markdown文件切分为chunk
4. 使用BM25算法作为第一条召回通道
5. 使用[gme模型](https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-2B-Instruct)将多模态chunk编码为embedding，并使用[faiss](https://github.com/facebookresearch/faiss)进行向量检索作为第二条召回通道
6. 使用[jina模型](https://huggingface.co/jinaai/jina-reranker-m0)作为多模态的reranker，融合多种召回结果

AGENT实现：
1. 构建SQL查询的Function Calling MCP
2. [Qwen VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)生成SQL语句查询用户的标签信息与投资行为序列作为个性化Query
3. 使用Query进行RAG检索，得到用户可能偏好的投资信息
4. 使用Qwen VL作为多模态模型给出投资建议
5. 构建Email发送的Function Calling MCP
6. Qwen VL 调用Email发送的Function，给用户发送当日投资建议的邮件

环境安装：
```shell
pip install -r requirements.txt 
```

#### 爬取知识库
爬虫+PDF解析

#### 预处理chunk
chunk的格式为：
```json lines
[
    {"type":"text","text":"context 1"},
    {"type":"image","image":"image path/url 1"},
    {"type":"text","text":"context 2"},
    {"type":"image","image":"image path/url 2"},
]
```
这样的一行json可以被转化为markdown文本：
```markdown
context 1 
![](image path/url 1)  
context 2
![](image path/url 2)
```
