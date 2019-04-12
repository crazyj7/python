'''
naver top 10 news
'''


import codecs
import urllib.request
from bs4 import BeautifulSoup
import json

url = 'http://m.naver.com'

response = urllib.request.urlopen(url)
# data = response.read()
# data2 = codecs.utf_8_decode(data)
# print('respose=', data2)

soup = BeautifulSoup(response, "html.parser")
print(soup)
''' div id=RTK_LIST,    span cllass=cd_t 
'''
rank = soup.select("div #RTK_LIST")
print(rank)

