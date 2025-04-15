import os
import sys
import re
import glob

from natsort import natsorted
from tqdm import tqdm
import jsonlines
from langchain_text_splitters import MarkdownHeaderTextSplitter
from loguru import logger

def verify_chunk(chunk):
    """使用chunk重新组装出原来的文本"""
    markdown_text=""
    for item in chunk:
        if item['type']=="text":
            markdown_text=markdown_text+item['text']
        else:
            markdown_text=markdown_text+f"![]({item['image']})"
    return markdown_text

def convert_chunk(markdown_text):
    # 匹配 Markdown 图片语法的正则表达式
    pattern = r"!\[.*?\]\(.*?\)"
    # 使用正则表达式找到所有图片位置
    split_points = [(m.start(),m.end()) for m in re.finditer(pattern, markdown_text)]
    text_end=0
    chunk=[]
    for split_point in split_points:
        image_start,image_end=split_point
        if text_end<image_start:
            text=markdown_text[text_end:image_start]
            chunk.append({"type":"text","text":text})
            text_end=image_end
        image=markdown_text[image_start:image_end]
        image_path=re.search(r'\!\[\]\((.*?)\)', image).group(1) # 从 ![](image_path) 中提取 image_path
        chunk.append({"type":"image","image":image_path})
        # if image!=f"![]({image_path})":
        #     import pdb;pdb.set_trace()
        
    if text_end<len(markdown_text)-1:
        text=markdown_text[text_end:len(markdown_text)]  
        chunk.append({"type":"text","text":text})
    # combined_markdown_text=verify_chunk(chunk)
    # if markdown_text!=combined_markdown_text:
    #     import pdb;pdb.set_trace()
    return  chunk

def get_chunks(markdown_path,abs_path=True,min_length=100):
    chunks=[]
    with open(markdown_path,'r') as f:
        markdown_document=f.read()
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(markdown_document)
    for md_header_split in md_header_splits:
        t=md_header_split.to_json()['kwargs']
        markdown_text="\n".join([f"{k}: {v}" for k,v in t.get('metadata',dict()).items()])+f"\n {t['page_content']}"
        if len(markdown_text)<min_length:
            continue
        chunk=convert_chunk(markdown_text)
        if abs_path: # 将相对地址转换为绝对地址
            markdown_folder=os.path.dirname(markdown_path)
            for i in range(len(chunk)):
                if chunk[i]['type']=='image':
                    chunk[i]['image']=os.path.join(markdown_folder,chunk[i]['image'])
        chunks.append(chunk)
    return chunks
    
    
if __name__=="__main__":
    folder="/data/wzh_fd/workspace/tiny-mm-rag-agent/data/dong_fang_cai_fu/outputs"
    chunks=[]
    for md_file in tqdm(natsorted(glob.glob(os.path.join(folder,"*")))):
        markdown_path=os.path.join(md_file,'ocr',md_file.split('/')[-1]+'.md')
        chunks+=get_chunks(markdown_path)
    logger.info(f"=> 共产生 {len(chunks)} 个 chunk块 ..")
    with jsonlines.open(os.path.join("/data/wzh_fd/workspace/tiny-mm-rag-agent/data/dong_fang_cai_fu","chunks.json"),'w') as writer:
        for chunk in chunks:
            writer.write(chunk)