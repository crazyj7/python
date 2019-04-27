'''
google image downloader
google web page ; search and copy URL.

'''

from bs4 import BeautifulSoup
import urllib.request
import time
from selenium import webdriver
import codecs
import os
import json

url = 'https://www.google.co.kr/search?hl=ko&authuser=0&tbm=isch&source=hp&biw=1536&bih=722&ei=94zBXI2hCrSOr7wP-LSh6AM&q=%EA%B0%90%EC%9D%80+%EB%88%88&oq=%EA%B0%90%EC%9D%80+%EB%88%88&gs_l=img.3..0l3j0i5i30l6j0i8i30.2001.3167..3407...0.0..0.110.1000.5j5......0....1..gws-wiz-img.....0..0i24.GRQBRRTjwzk'
savedir='download/eye'
if not os.path.exists(savedir):
    os.mkdir(savedir)


options = webdriver.ChromeOptions()
options.add_argument('headless')
browser = webdriver.Chrome(chrome_options=options)
browser.implicitly_wait(3)
browser.get(url)
browser.save_screenshot("web1.png")

html = browser.page_source
soup = BeautifulSoup(html, 'lxml')
# print(soup)
# html.find_all("")
metas = soup.select("div.rg_meta.notranslate")
for m in metas:
    # print(m)      ## <div class="rg_meta notranslate">{"cl":21,"cr":9,"dhl... "ou":"..."
    print(m.text)
    jj = json.loads(m.text)
    print(jj["ou"])


