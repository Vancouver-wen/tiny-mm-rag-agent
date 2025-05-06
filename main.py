import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Disabling parallelism to avoid deadlocks...
import argparse
import json
import asyncio
from pprint import pprint
import re

import torch
import yaml
import jsonlines
from easydict import EasyDict
from loguru import logger
from tqdm import tqdm
from joblib import Parallel,delayed
from fastmcp import Client
from fastmcp.client import SSETransport

from tools.llm.qwenvl import QwenVL,RemoteQwenVL
from tools.searcher.searcher import Searcher
from tools.agent.prompt import Prompt

class TinyRAG:
    def __init__(self, config,build=False) -> None:
        logger.info(f"config: {config}")
        self.config = config
        self.searcher = Searcher(
            base_dir=os.path.dirname(self.config.chunk_file),
            embedding_model=self.config.embedding_model,
            embedding_dim=self.config.embedding_dim,
            reranker_model=self.config.reranker_model,
            device_retriever="cuda:0",
            device_reranker="cuda:0",
            is_remote=self.config.remote,
            remote_embedding_model=self.config.remote_embedding_model,
            remote_reranker_model=self.config.remote_reranker_model,
        )
        if self.config.remote:
            self.llm=RemoteQwenVL(self.config.remote_llm_path)
        else:
            self.llm = QwenVL(
                model_path=self.config.llm_model,
                device="balanced_low_0"
            )
        self.prompt=Prompt(self.config)
        if build:
            chunks=[]
            chunk_file=self.config.chunk_file
            with jsonlines.open(chunk_file,'r') as reader:
                for obj in reader:
                    chunks.append(obj)
            # chunks=chunks[:100] # fast debug
            self.build(chunks)
        else:
            self.load()

    def build(self, chunks:list[dict]):
        """ 构建数据库可能需要很长时间"""
        self.searcher.build_db(chunks)
        self.searcher.save_db()

    def load(self):
        self.searcher.load_db()

    def chat(self) -> list:
        """
        一开始不要想着自动化，先手动控制流程
        """
        prompt=self.prompt.get_prompt()
        messages = [
            {
                "role":"system",
                "content":[
                    {"type":"text","text":prompt}
                ]
            },
            {
                "role": "user",
                "content": [
                    # {"type": "image","image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",},
                    {"type": "text", "text": "请你给出我一些投资建议。"},
                ],
            }
        ]
        # LLM的初次回答
        logger.info(f"query 改写 ..")
        response = self.llm.generate(messages)
        import pdb;pdb.set_trace()
        messages.append({
            'role':'assistent',
            'content':[{'type':'text','text':response}]
        })
        # 数据库检索的文本
        ## 拼接 query和LLM初次生成的结果，查找向量数据库
        chunk=messages[0]['content']+messages[1]['content']
        search_content_list = self.searcher.search(chunk, top_n=self.config.top_n) # list of (score,chunk)
        # 构造 输入
        messages.append({
            'role':"user",
            'content':[{'type':'text','text':'--- \n参考信息: \n'}]
        })
        for score,chunk in search_content_list:
            messages[-1]['content']=messages[-1]['content']+chunk
        messages[-1]['content'].append({
            'type':'text',
            'text':"""
---
请根据上述参考信息回答，修正之前的回答。
前面的参考信息和回答可能有用，也可能没用，你需要从我给出的参考信息中选出与我的问题最相关的那些，来为你修正的回答提供依据。
你修正的回答:
"""
        })
        # 生成最终答案
        logger.info(f"query 增强推理 ...")
        try:
            response = self.llm.generate(messages)
            messages.append({
                'role':'assistent',
                'content':[{'type':'text','text':response}]
            })
            logger.info(f"messages have been saved ..")
        except Exception as e:
            logger.info(f"enter error : {e}")

        return messages

class TinyAgenticRAG:
    def __init__(self,config):
        logger.info(f"config: {config}")
        self.config = config
        if self.config.remote:
            self.llm=RemoteQwenVL(self.config.remote_llm_path)
        else:
            self.llm = QwenVL(
                model_path=self.config.llm_model,
                device="balanced_low_0"
            )
        self.prompt=Prompt(
            memobase_token=self.config.memobase_token,
            memobase_uid=self.config.memobase_uid
        )
    
    async def chat(self) -> list:
        """
        一开始不要想着自动化，先手动控制流程
        """
        clients = {
            mcpServer:Client(SSETransport(url=self.config.mcpServers[mcpServer].url)) 
            for mcpServer in self.config.mcpServers
        }
    
        prompt=await self.prompt.get_prompt(clients)
        
        messages = [
            {
                "role":"system",
                "content":[
                    {"type":"text","text":prompt}
                ]
            },
            {
                "role": "user",
                "content": [
                    # {"type": "image","image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",},
                    {"type": "text", "text": """请你给出我一些投资建议。
你要将我的问题当作query，使用agentic rag mcp中的rewrite query来改写query，使用search query来逐一查询改写后的query来获取更多的相关信息。
然后综合利用所有的信息来回答原始的query。
"""},
                ],
            }
        ]
        while True:
            response = self.llm.generate(messages)
            messages.append({
                'role':'assistant',
                'content':[{'type':'text','text':response}]
            })
            print(response)
            # 分析是否需要调用工具 
            # 调用工具 / input
            pattern = r"<use_mcp_tool>(.*?)</use_mcp_tool>"
            pts = re.findall(pattern, response, re.DOTALL)
            if pts: # 有工具调用
                mcp_use=pts[0]
                server_name=re.findall(r"<server_name>(.*?)</server_name>", mcp_use, re.DOTALL)[0]
                tool_name=re.findall(r"<tool_name>(.*?)</tool_name>", mcp_use, re.DOTALL)[0]
                arguments=re.findall(r"<arguments>(.*?)</arguments>", mcp_use, re.DOTALL)[0]
                try:
                    import pdb;pdb.set_trace()
                    arguments=json.loads(arguments)
                    async with clients[server_name] as client:
                        assert client.is_connected()
                        tool_response = await client.call_tool_mcp(name=tool_name,arguments=arguments)
                        tool_response = tool_response.content[0].text
                        tool_response = f"[use_mcp_tool for '{tool_name}' in '{server_name}'] Result: \n {tool_response}"
                except:
                    tool_response=f"mcp tool: {tool_name} call failed. \n"
                messages.append({
                    'role':'user',
                    'content':[{'type':'text','text':tool_response}]
                })
            else:
                logger.info(f"请用户输入：")
                user_input=input()
                messages.append({
                    'role':'user',
                    'content':[{'type':'text','text':user_input}]
                })
                if user_input=='exit':
                    break
            with open("./log.json",'w',encoding="UTF-8") as f:
                json.dump(messages,f,ensure_ascii=False,indent=4)
        return messages
        

async def main(args):
    with open(args.config,'r', encoding='utf-8') as f:
        config=EasyDict(yaml.safe_load(f))
        
    # tiny_rag = TinyRAG(config,args.build)
    tiny_rag = TinyAgenticRAG(config)
    
    # 这里可以测试 llm 的任务拆解能力，以及多步执行能力
    messages=await tiny_rag.chat() 
    with open("./log.json",'w',encoding="UTF-8") as f:
        json.dump(messages,f,ensure_ascii=False,indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tiny RAG Argument Parser')
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--config', type=str, default="config/config.yaml", help='Tiny RAG config')

    args = parser.parse_args()
    
    asyncio.run(main(args))

"""
请你给出我一些投资建议。\
你要将我的问题当作query，使用agentic rag mcp中的rewrite query来改写query，使用search query来逐一查询改写后的query来获取更多的相关信息。\
然后综合利用所有的信息来回答原始的query。


最后，我授予你直接访问数据库的权限。你需要使用sql_mcp在数据库执行查询语句，根据我的姓名找到我的邮箱地址，使用email_mcp将你的投资建议发送到我的邮箱中。\

"""