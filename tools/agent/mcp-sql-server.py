import os
import sys
import json
import httpx
import argparse
from typing import Any
if __name__=="__main__":
    sys.path.append(".")

import yaml
import sqlite3
from easydict import EasyDict
from loguru import logger
from fastmcp import FastMCP
from openai import OpenAI
from tools.searcher.searcher import Searcher

# 初始化 MCP 服务器
mcp = FastMCP("SqlServer",host="127.0.0.1", port=9001)

class DataBase(object):
    def __init__(self,db_path="./example.db"):
        path_exist = os.path.exists(db_path)
        self.con = sqlite3.connect(db_path)
        if not path_exist:
            logger.info(f"do not find db file, create it !")
            self.create_example()
        self.cur = self.con.cursor()
    
    def create_example(self,):
        # Create user table
        self.cur.execute('''CREATE TABLE user_info (name, email)''')
        # Insert a row of data
        self.cur.execute("INSERT INTO user_info VALUES ('WZH','1052951572@qq.com')")
        self.cur.execute("INSERT INTO user_info VALUES ('CJ','22210240312@m.fudan.edu.cn')")
        # Save (commit) the changes
        self.con.commit()
        
        # Create user table
        self.cur.execute("""CREATE TABLE user_investment (
id INT AUTO_INCREMENT PRIMARY KEY,
user_id VARCHAR(50) NOT NULL,
investment VARCHAR(100),
investment_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")
        # Insert a row of data
        self.cur.execute("""INSERT INTO user_investment (user_id, investment, investment_time) VALUES
('WZH', '特斯拉', '2025-05-01 10:00:00'),
('WZH', '纳斯达克指数', '2025-05-02 11:00:00'),
('WZH', '茅台', '2025-05-03 12:00:00'),
('WZH', '五粮液', '2025-05-04 13:00:00'),
('CJ', '光伏产业', '2025-04-18 06:00:00'),
('CJ', '宁德时代', '2025-04-12 11:00:00'),
('CJ', '可口可乐', '2025-04-22 22:00:00'),
('CJ', '福耀玻璃', '2025-04-04 19:00:00');
""")
        # Save (commit) the changes
        self.con.commit()

    def __del__(self,):
        # super().__del__()
        # We can also close the connection if we are done with it.
        # Just be sure any changes have been committed or they will be lost.
        self.con.close()

parser = argparse.ArgumentParser(description='Mcp Sql Server Argument Parser')
parser.add_argument('--config', type=str, default="/data/wzh_fd/workspace/tiny-mm-rag-agent/config/config.yaml", help='Tiny RAG config')

args = parser.parse_args()
with open(args.config,'r') as f:
    config = yaml.safe_load(f)

config = EasyDict(config)
data_base=DataBase(config.mcpServers.sql_mcp.db_path)

@mcp.tool()
def get_table_structures(*args,**kwargs)->dict[str:list]:
    """
    查询数据库中包含的所有表以及每个表的表结构信息。
    :return: 一个字典，键为表名，值为该表的结构信息列表。每个结构信息是一个字典，包含列的详细信息。
    """
    try:
        # 连接到 SQLite 数据库
        cursor = data_base.cur

        # 获取数据库中所有表的名称
        cursor.execute("""
SELECT name
FROM sqlite_master
WHERE type = 'table'
ORDER BY name;
""")
        tables = cursor.fetchall()

        # 存储每个表的结构信息
        table_structures = {}

        # 遍历每个表，获取其结构信息
        for table in tables:
            table_name = table[0]
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()

            # 将列信息存储为字典列表
            table_structures[table_name] = [
                {
                    "cid": col[0],  # 列的序号
                    "name": col[1],  # 列名
                    "type": col[2],  # 数据类型
                    "notnull": col[3],  # 是否允许为空
                    "dflt_value": col[4],  # 默认值
                    "pk": col[5]  # 是否为主键
                }
                for col in columns
            ]

        return table_structures

    except sqlite3.Error as e:
        print(f"SQLite 错误: {e}")
        return {}

# @mcp.tool()
# def get_all_tables()->list[str]:
#     """
#     查询 SQLite 数据库中包含的所有表。
#     :return: 数据库中所有表以及每个表的结构。
#     """
#     try:
#         # 查询所有表
#         data_base.cur.execute("""
# SELECT name
# FROM sqlite_master
# WHERE type = 'table'
# ORDER BY name;
# """)
#         # 获取查询结果
#         tables = data_base.cur.fetchall()
#         # 提取表名并返回
#         table_names = [table[0] for table in tables]
#         return table_names
#     except sqlite3.Error as e:
#         print(f"SQLite 错误: {e}")
#         return []

# @mcp.tool()
# def get_table_structures(table_name)->list[dict]:
#     """
#     查询 SQLite 数据库中某个表的表结构信息。
#     注意！在使用该函数之前，要先使用 get_all_tables方法获取 SQLite 数据库中包含的所有表的table_name，然后才能使用该方法查询某个已存在的表的表结构信息。
#     :param table_name: SQLite 数据库中的某个表的表名称。
#     :return: 该表的结构信息列表。每个结构信息是一个字典，包含列的详细信息。
#     """
#     try:
#         # 查询所有表
#         data_base.cur.execute(f"PRAGMA table_info({table_name})")
#         # 获取查询结果
#         columns = data_base.cur.fetchall()
#         # 提取表结构并返回
#         table_structure = [
#             {
#                 "colume_id": col[0],  # 列的序号
#                 "name": col[1],  # 列名
#                 "type": col[2],  # 数据类型
#                 "not_null": col[3],  # 是否允许为空
#                 "default_value": col[4],  # 默认值
#                 "is_primary_key": col[5]  # 是否为主键
#             }
#             for col in columns
#         ]
#         return table_structure
#     except sqlite3.Error as e:
#         print(f"SQLite 错误: {e}")
#         return []

@mcp.tool()
def execute_select_sql(sql_statement:str)->list[tuple]:
    """
    在数据库中执行一条 SQLite 查询语句。
    :param sql_statement: 要执行的 SQL 查询语句。一条正确的SQL查询语句应该以 SELECT 开头，例如 SELECT * FROM table_name WHERE condition
    :return: 查询结果。
    
    注意! 在调用该函数之前，必须先调用 get_table_structures 工具 获取表结构信息。你必须遵守这个规定，否则将会编写出错误的sql_statement。
    Important! Before calling this function, you must first call the get_table_structures tool to obtain the table structure information. You must follow this rule; otherwise, you will end up writing an incorrect sql_statement.
    """
    # When to use this tool:
    # - 从 memory tag 中已知用户的姓名等信息，但还需要更多用户的行为数据
    # - 用户要求从数据库中查询信息
    # - 用户当前的上下文信息不足，需要补充更多的用户信息
    # Key features:
    # - 该函数已经被系统管理员授予了直接访问数据库的权限。
    # - 该函数内部已经建立了实际的数据库连接，可以直接执行SQL查询。
    # - 该函数具有直接访问数据库的能力，通过调用该函数可以从数据库中获取信息。
    try:
        data_base.cur.execute(sql_statement)
        # 如果是 SELECT 查询，获取结果
        if sql_statement.strip().upper().startswith("SELECT"):
            result = data_base.cur.fetchall()
        else:
            result = f"sql_statement: {sql_statement} 中不包含 select 关键词！因此拒绝执行该语句。"
        return result
    except sqlite3.Error as e:
        return f"SQLite 错误: {e}"

        
if __name__=="__main__":
    
    # for row in data_base.cur.execute('SELECT * FROM user_info WHERE name=\'CJ\''):
    #     print(row)
    # for row in data_base.cur.execute("PRAGMA table_info(user_info);"):
    #     print(row)
    # tables=get_all_tables()
    # for table in tables:
    #     table_structure=get_table_structures(table_name=table)
    #     logger.info(f"{table}: \n {table_structure}")
    # result=execute_select_sql('SELECT * FROM user_info WHERE name=\'CJ\'')
    # import pdb;pdb.set_trace()
    
    # Default: runs stdio transport
    # mcp.run()

    # Example: Run with SSE transport on a specific port
    mcp.run(transport="sse")