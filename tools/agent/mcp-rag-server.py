import os
import sys
import json
import httpx
from typing import Any
if __name__=="__main__":
    sys.path.append(".")

import yaml
from easydict import EasyDict
from fastmcp import FastMCP
from openai import OpenAI
from tools.searcher.searcher import Searcher

# 初始化 MCP 服务器
mcp = FastMCP("RagServer",host="127.0.0.1", port=9000)

openai_api_base = "https://api.deepseek.com"
openai_api_key = "sk-102b49bc5d0249ec8ba89d22c0818c8d"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
SYSTEM_PROMPT = """
用户将会执行一个query改写任务。 你会将一个query改写为多个querys. Please parse the 改写后的querys into JSON format. 

EXAMPLE INPUT: 
冬日四件套.

EXAMPLE JSON OUTPUT:
{
    "rewritten_queries":[
        "冰糖葫芦",
        "烤地瓜",
        "炒栗子",
        "热奶茶",
    ]
}
"""

with open("/data/wzh_fd/workspace/tiny-mm-rag-agent/config/config.yaml",'r', encoding='utf-8') as f:
    config=EasyDict(yaml.safe_load(f))

searcher = Searcher(
    base_dir=os.path.dirname(config.chunk_file),
    embedding_model=config.embedding_model,
    embedding_dim=config.embedding_dim,
    reranker_model=config.reranker_model,
    device_retriever="cuda:0",
    device_reranker="cuda:0",
    is_remote=config.remote,
    remote_embedding_model=config.remote_embedding_model,
    remote_reranker_model=config.remote_reranker_model,
)

searcher.load_db()

@mcp.tool()
def rewrite_query(query:str,information:str,num:int=2)->list[str]:
    """
    能够根据information来改写query。为了改善用户体验，应当尽可能从memory tag、数据库与当前对话上下文中得到information，减少需要用户手动输入information的场景。
    :param query: 用于在检索相关信息的query
    :param information: 用于改写query的上下文信息，主要是用户画像、与query可能相关的历史聊天记录等。
    :return: 一个包含若干改写后query的列表
    """
    # information应该尽可能详细丰富，否则将无法有效改写query。
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role": "user", "content": f"""请你根据 **information** 的内容对 **query** 进行查询改写与查询拓展，并以json格式返回一个长度为{num}的列表。
查询改写的应用方式是对原始Query拓展出与用户需求关联度高的改写词，多个改写词与用户搜索词一起做检索。
查询改写主要用于解决一些语义鸿沟，比如语义拓展（同义词，下位词，大小写，繁简转换等）、场景拓展、口语与专业术语的Gap、意图识别等。
查询改写应该对**information**进行凝念概括与场景拓展，从而进行更好的挖掘用户**query**中的潜在语义。
**information**: {information} 
**query**: {query}
"""},
        ],
        response_format={
            'type': 'json_object'
        }
    )
    r=json.loads(response.choices[0].message.content)
    try:
        assert isinstance(r,dict)
        assert r.__contains__("rewritten_queries")
        r=r['rewritten_queries']
        assert isinstance(r,list)
    except:
        r=None
    return r

@mcp.tool()
def search_query(query:str,num:int=2)->list[dict]:
    """
    能够根据query在知识库中检索若干条相关的信息。
    通常情况下，在使用该方法前需要先进行query改写，得到多个改写后的query。然后多次调用该方法来查询每一个改写后的query对应的相关信息。
    :param query: 用于检索的查询
    :param num: 检索到的相关信息的条数。尽量维持一个比较小的检索条数，否则会超过上下文最大长度限制
    :return: 一个列表包含若干检索到的相关信息，每个信息都是一个字典
    """
    chunk=[{'type':'text','text':query}]
    chunks=searcher.search(chunk, top_n=num)
    result=[]
    for score,cs in chunks:
        for c in cs:
            result.append(c)
    return result

if __name__=="__main__":
    query="请你给我一些投资建议"
    information="""<memory>
# Below is the user profile:
- work::industry: 用户对技术行业有一定了解和关注，关注新能源汽车和人工智能领域的技术动态
- basic_info::name: User's name is WZH
- basic_info::age: 25
- interest::technology: 用户对英伟达公司的技术优势和商业壁垒感兴趣
- interest::automobile: 用户对小米SU7和比亚迪汉这两款新能源汽车感兴趣，关注续航里程和智能驾驶辅助系统
- psychological::motivations: 用户希望获取最新的技术动态和市场信息
- interest::value: 用户在意新能源汽车的续航里程、智能驾驶辅助系统和性价比

# Below is the latest events of the user:
2025/05/01:
用户询问了小米SU7和比亚迪汉新能源汽车的配置和性价比，助理详细介绍了两款车的续航、智能驾驶辅助系统及性价比，并表示会关注英伟达的最新动态。

---
2025/04/25:
用户询问了小米SU7和比亚迪汉的新能源汽车配置，特别关注续航和智能驾驶辅助系统，助手详细介绍了两款车的特点，并讨论了英伟达的商业壁垒和技术优势。用户还请求关注英伟达的最新动态。

</memory>
"""
    # querys=rewrite_query(query,information)
    # for query in querys:
    #     chunks=search_query(query,num=6)
    #     import pdb;pdb.set_trace()
    
    
    # Default: runs stdio transport
    # mcp.run()

    # Example: Run with SSE transport on a specific port
    mcp.run(transport="sse")