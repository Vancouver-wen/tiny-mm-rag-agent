"""
构建 mcp client
"""

import asyncio

from fastmcp import Client
from fastmcp.client import SSETransport


async def main():
    async with Client(SSETransport(url='http://localhost:9000/sse')) as client:
        print(f'client is connected ? {client.is_connected()}')
        tools = await client.list_tools()
        print(f'available tools: {tools}')


if __name__ == '__main__':
    asyncio.run(main())