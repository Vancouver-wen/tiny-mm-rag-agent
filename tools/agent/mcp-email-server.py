import os
import sys
import json
import httpx
from typing import Any
import argparse
import smtplib
from email.mime.text import MIMEText
from email.header import Header
if __name__=="__main__":
    sys.path.append(".")

import yaml
from easydict import EasyDict
from fastmcp import FastMCP
from openai import OpenAI
from tools.searcher.searcher import Searcher

# 初始化 MCP 服务器
mcp = FastMCP("EmailServer",host="127.0.0.1", port=9002)

@mcp.tool()
def send_email(Context:str,ToAddress:str,Head="From Finance Agent: GePT")->str:
    """
    能够将Head与Context以邮件的方式发送到ToAddress邮箱。
    :param Context: 邮件的主体内容
    :param ToAddress: 一个邮箱地址，标准结构是 用户名@域名。邮件会被发送到该邮箱地址
    :param Head: 邮件的抬头标题，通常是邮件主体内容的高度概括
    :return: 一个表示邮箱是否发送成功的布尔变量。如果邮件发送失败，还会返回异常的原因。
    """
    try:
        # 创建 SMTP 对象
        smtp = smtplib.SMTP("smtp.qq.com", port=587)
        smtp.starttls()
        smtp.ehlo() # 发送hello消息
        # 构造MIMEText对象，参数为：正文，MIME的subtype，编码方式
        message = MIMEText(Context, 'plain', 'utf-8')
        message['From'] = Header("From Admin <1052951572@qq.com>")  # 发件人的昵称
        message['To'] = Header("zhwen", 'utf-8')  # 收件人的昵称
        message['Subject'] = Header(Head, 'utf-8')  # 定义主题内容
        print(message)
        # 登录，需要：登录邮箱和授权码
        smtp.login(user="1052951572@qq.com", password="ewfcqtbqxmbqbfhb")
        smtp.sendmail(
            from_addr="1052951572@qq.com", 
            to_addrs=ToAddress,  
            # to_addrs实际上应当是个list，意味着你可以同时给多个账户发送，
            # 如果是字符串则默认为长度为1的list
            msg=message.as_string()
            )
        smtp.quit()
        return True
    except Exception as e:
        return f"False. Exception: {e}"

if __name__== "__main__" :
    # Head="Function Calling Test"
    # Context="test Context"
    # sendemail(Head,Context,ToAddress="1052951572@qq.com")
    
    # Default: runs stdio transport
    # mcp.run()

    # Example: Run with SSE transport on a specific port
    mcp.run(transport="sse")