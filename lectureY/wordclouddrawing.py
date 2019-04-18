
from collections import Counter
import urllib
import random
import webbrowser

from konlpy.tag import Hannanum
from lxml import html

import pytagcloud # requires Korean font support
import sys
import codecs
from MyUtil import MyUtil

import pickle

'''
copy font...
NanumGothic.ttf to env directory's font.
 
 ex dir) /opt/anaconda3/envs/tensorflow/lib/python3.5/site-packages/pytagcloud/fonts#

append font.json 
{
                "name": "NanumGothic",
                "ttf": "NanumGothic.ttf",
                "web": "http://fonts.googleapis.com/earlyaccess/nanumgothic.css"
}


'''


if sys.version_info[0] >= 3:
    urlopen = urllib.request.urlopen
else:
    urlopen = urllib.urlopen


r = lambda: random.randint(0,255)
color = lambda: (r(), r(), r())

def get_bill_text(billnum):
    url = 'http://pokr.kr/bill/%s/text' % billnum
    response = urlopen(url).read().decode('utf-8')
    page = html.fromstring(response)
    text = page.xpath(".//div[@id='bill-sections']/pre/text()")[0]
    return text

def get_tags(text, ntags=50, multiplier=10):
    h = Hannanum(max_heap_size=1024*4)  # max memory 4GB
    # h = Hannanum(max_heap_size=1024*1024)
    nouns = h.nouns(text)
    count = Counter(nouns)
    return [{ 'color': color(), 'tag': n, 'size': c*multiplier }\
                for n, c in count.most_common(ntags)]

def draw_cloud(tags, filename, fontname='Noto Sans CJK', size=(1000, 1000)):
    pytagcloud.create_tag_image(tags, filename, fontname=fontname, size=size)
    webbrowser.open(filename)




########################################################

MyUtil.set_cwd()

# targetfile="toji01.txt"
# targetfile="toji01.wakati"
# output = "tojs01.png"

targetfile="mini_namu.wakati"
output="mini_namu.png"


if True:
    # bill_num = '1904882'
    # text = get_bill_text(bill_num)
    file = codecs.open(targetfile, 'r', encoding='utf-8')
    text = file.read()
    print('call get_tags')
    tags = get_tags(text, ntags=100, multiplier=1)       # multiplier too big!!!
    print('end get_tags')
    with open('wordcloud.pkl', 'wb') as f :
        pickle.dump(tags, f)
else:
    with open('wordcloud.pkl', 'rb') as f:
        tags = pickle.load(f)


print(len(tags))
print('tags=', tags)

###########################################################

# get only TOP10
newtags=[]
for i in range(20):
    if tags[i]['tag'] in ['것', '그', '이', '니', '기', '뒤', '리', '지'] :    # skip
        continue
    newtags.append(tags[i])

# font size adjust (or multipler change!)
for it in newtags:
    it['size']=int(it['size']/4)

# drawing
draw_cloud(newtags, output, fontname='NanumGothic')

print('create ok')
