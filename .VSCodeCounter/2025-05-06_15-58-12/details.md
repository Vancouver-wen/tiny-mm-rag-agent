# Details

Date : 2025-05-06 15:58:12

Directory /data/wzh_fd/workspace/tiny-mm-rag-agent

Total : 34 files,  2639 codes, 390 comments, 534 blanks, all 3563 lines

[Summary](results.md) / Details / [Diff Summary](diff.md) / [Diff Details](diff-details.md)

## Files
| filename | language | code | comment | blank | total |
| :--- | :--- | ---: | ---: | ---: | ---: |
| [tiny-mm-rag-agent/README.md](/tiny-mm-rag-agent/README.md) | Markdown | 71 | 0 | 12 | 83 |
| [tiny-mm-rag-agent/config/config.yaml](/tiny-mm-rag-agent/config/config.yaml) | YAML | 20 | 0 | 7 | 27 |
| [tiny-mm-rag-agent/main.py](/tiny-mm-rag-agent/main.py) | Python | 212 | 12 | 22 | 246 |
| [tiny-mm-rag-agent/preprocessing/chunk.py](/tiny-mm-rag-agent/preprocessing/chunk.py) | Python | 70 | 7 | 7 | 84 |
| [tiny-mm-rag-agent/preprocessing/parsePDF.sh](/tiny-mm-rag-agent/preprocessing/parsePDF.sh) | Shell Script | 36 | 72 | 28 | 136 |
| [tiny-mm-rag-agent/preprocessing/parser/__init__.py](/tiny-mm-rag-agent/preprocessing/parser/__init__.py) | Python | 30 | 1 | 10 | 41 |
| [tiny-mm-rag-agent/preprocessing/parser/base_parser.py](/tiny-mm-rag-agent/preprocessing/parser/base_parser.py) | Python | 31 | 5 | 8 | 44 |
| [tiny-mm-rag-agent/preprocessing/parser/doc_parser.py](/tiny-mm-rag-agent/preprocessing/parser/doc_parser.py) | Python | 56 | 2 | 15 | 73 |
| [tiny-mm-rag-agent/preprocessing/parser/img_parser.py](/tiny-mm-rag-agent/preprocessing/parser/img_parser.py) | Python | 25 | 1 | 10 | 36 |
| [tiny-mm-rag-agent/preprocessing/parser/md_parser.py](/tiny-mm-rag-agent/preprocessing/parser/md_parser.py) | Python | 61 | 3 | 21 | 85 |
| [tiny-mm-rag-agent/preprocessing/parser/pdf_parser.py](/tiny-mm-rag-agent/preprocessing/parser/pdf_parser.py) | Python | 102 | 4 | 23 | 129 |
| [tiny-mm-rag-agent/preprocessing/parser/ppt_parser.py](/tiny-mm-rag-agent/preprocessing/parser/ppt_parser.py) | Python | 57 | 2 | 15 | 74 |
| [tiny-mm-rag-agent/preprocessing/parser/txt_parser.py](/tiny-mm-rag-agent/preprocessing/parser/txt_parser.py) | Python | 56 | 3 | 17 | 76 |
| [tiny-mm-rag-agent/preprocessing/reptile.py](/tiny-mm-rag-agent/preprocessing/reptile.py) | Python | 81 | 18 | 9 | 108 |
| [tiny-mm-rag-agent/preprocessing/sentence_splitter.py](/tiny-mm-rag-agent/preprocessing/sentence_splitter.py) | Python | 55 | 4 | 5 | 64 |
| [tiny-mm-rag-agent/requirements.txt](/tiny-mm-rag-agent/requirements.txt) | pip requirements | 25 | 3 | 1 | 29 |
| [tiny-mm-rag-agent/server.py](/tiny-mm-rag-agent/server.py) | Python | 63 | 7 | 20 | 90 |
| [tiny-mm-rag-agent/tools/__init__.py](/tiny-mm-rag-agent/tools/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [tiny-mm-rag-agent/tools/agent/__init__.py](/tiny-mm-rag-agent/tools/agent/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [tiny-mm-rag-agent/tools/agent/mcp-client.py](/tiny-mm-rag-agent/tools/agent/mcp-client.py) | Python | 13 | 0 | 6 | 19 |
| [tiny-mm-rag-agent/tools/agent/mcp-email-server.py](/tiny-mm-rag-agent/tools/agent/mcp-email-server.py) | Python | 47 | 12 | 6 | 65 |
| [tiny-mm-rag-agent/tools/agent/mcp-rag-server.py](/tiny-mm-rag-agent/tools/agent/mcp-rag-server.py) | Python | 114 | 11 | 18 | 143 |
| [tiny-mm-rag-agent/tools/agent/mcp-sql-server.py](/tiny-mm-rag-agent/tools/agent/mcp-sql-server.py) | Python | 118 | 80 | 25 | 223 |
| [tiny-mm-rag-agent/tools/agent/memo.py](/tiny-mm-rag-agent/tools/agent/memo.py) | Python | 115 | 0 | 6 | 121 |
| [tiny-mm-rag-agent/tools/agent/prompt.py](/tiny-mm-rag-agent/tools/agent/prompt.py) | Python | 244 | 21 | 56 | 321 |
| [tiny-mm-rag-agent/tools/llm/__init__.py](/tiny-mm-rag-agent/tools/llm/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [tiny-mm-rag-agent/tools/llm/deepseek.py](/tiny-mm-rag-agent/tools/llm/deepseek.py) | Python | 43 | 8 | 16 | 67 |
| [tiny-mm-rag-agent/tools/llm/qwenvl.py](/tiny-mm-rag-agent/tools/llm/qwenvl.py) | Python | 78 | 21 | 13 | 112 |
| [tiny-mm-rag-agent/tools/searcher/__init__.py](/tiny-mm-rag-agent/tools/searcher/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [tiny-mm-rag-agent/tools/searcher/gme_inference.py](/tiny-mm-rag-agent/tools/searcher/gme_inference.py) | Python | 256 | 31 | 44 | 331 |
| [tiny-mm-rag-agent/tools/searcher/reanker.py](/tiny-mm-rag-agent/tools/searcher/reanker.py) | Python | 127 | 11 | 21 | 159 |
| [tiny-mm-rag-agent/tools/searcher/retriever_bm25.py](/tiny-mm-rag-agent/tools/searcher/retriever_bm25.py) | Python | 189 | 18 | 44 | 251 |
| [tiny-mm-rag-agent/tools/searcher/retriever_embed.py](/tiny-mm-rag-agent/tools/searcher/retriever_embed.py) | Python | 173 | 19 | 30 | 222 |
| [tiny-mm-rag-agent/tools/searcher/searcher.py](/tiny-mm-rag-agent/tools/searcher/searcher.py) | Python | 71 | 14 | 15 | 100 |

[Summary](results.md) / Details / [Diff Summary](diff.md) / [Diff Details](diff-details.md)