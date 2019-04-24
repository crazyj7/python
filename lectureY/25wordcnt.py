import codecs
from bs4 import BeautifulSoup
from konlpy.tag import Okt

file = codecs.open('toji01.txt', 'r', encoding='utf-8')
text = file.read()

twitter = Okt()
word_dic = {}
lines = text.split('\r\n')
for line in lines:
    malist = twitter.pos(line)
    for taeso, pumsa in malist:
        if pumsa=='Noun':
            if not (taeso in word_dic):
                word_dic[taeso]=0
            word_dic[taeso]+=1
print(word_dic)
keys = sorted(word_dic.items(), key=lambda  x:x[1], reverse=True)
for word, count in keys[:50]:
    print(word, count, end=" ")

