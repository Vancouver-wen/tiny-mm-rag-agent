import os
import io
import requests
import time
import random
import urllib
import json

import wget
from tqdm import tqdm
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import StaleElementReferenceException, TimeoutException

options = webdriver.ChromeOptions()
# options.add_argument("--headless")  # 启用无头模式
driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)

# 打开网页
driver.get('https://data.eastmoney.com/report/industry.jshtml')
# 等待页面加载完成
wait = WebDriverWait(driver, 3)
wait.until(EC.presence_of_element_located((By.ID, "industry_table")))

# 定义关闭广告的函数
def close_ad():
    try:
        # 定位关闭广告的按钮
        close_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//img[@src='https://emcharts.dfcfw.com/fullscreengg/ic_close.png']")))
        close_button.click()
        print("关闭广告成功")
    except TimeoutException:
        print("未检测到广告弹窗")
    except Exception as e:
        print("关闭广告失败:", e)

# 定义翻页函数
def next_page():
    try:
        # 检测并关闭广告
        close_ad()
        # 定位“下一页”按钮并点击
        next_page_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//a[text()='下一页']")))
        next_page_button.click()
        # 等待新页面加载完成
        wait.until(EC.presence_of_element_located((By.ID, "industry_table")))
        print("翻到下一页")
    except Exception as e:
        print("没有更多页面或翻页失败:", e)

# 获取PDF下载链接
def get_pdf_url(url):
    try:
        res = requests.get(url) 
        soup = BeautifulSoup(res.text,'html.parser') 
        t=soup.find(class_="pdf-link")
        return t.get('href')
    except:
        return None

# 下载pdf文件
def download_pdf(url:str):
    try:
        pdf_name=url.split('?')[-1]
        wget.download(url, f'./reports/{pdf_name}')
        return pdf_name
    except:
        return None
hash_table=dict()
for i in tqdm(range(50)): # 爬取50页面
    print(f"当前页面：{i}")
    # 打印页面标题
    # print(driver.title)
    div_element = driver.find_element(By.ID,"industry_table")
    table_element=div_element.find_element(By.TAG_NAME, "table")
    # 获取表格的所有行
    rows = table_element.find_elements(By.TAG_NAME, "tr")
    row_datas=[]
    urls=[]
    # 遍历每一行，提取数据
    for row in rows:
        cells = row.find_elements(By.TAG_NAME, "td")  # 获取单元格
        links = row.find_elements(By.TAG_NAME, "a")
        if not cells: continue  # 如果有单元格（跳过表头）
        row_data = [cell.text for cell in cells][4]
        url = [link.get_attribute("href") for link in links][5]
        row_datas.append(row_data)
        urls.append(url)
    # print(f"{row_datas} : {urls}")
    # 这里保存好映射，到服务器上下载
    for row_data,url in list(zip(row_datas,urls)):
        hash_table[row_data]={
            'url_origin':url,
            'url_download':get_pdf_url(url)
        }
    next_page()
    time.sleep(5)

# 关闭浏览器
driver.quit()

with open("./files.json",'w',encoding="UTF-8") as f:
    json.dump(hash_table,f,indent=4,ensure_ascii=False)