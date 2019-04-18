'''
기사 분류기
나무위키로 학습한 워드 벡터를 이용.
주제별 단어와 기사의 유사성을 측정하여 max 분류로 한다.
기사와의 유사성 측정은?
기사내의 단어(조사 등 제외) 분포를 찾아 의미 있는 것을 찾아,스코어화.
'''
import codecs
from bs4 import BeautifulSoup
from konlpy.tag import Okt
from gensim.models import word2vec
import time
import numpy as np

category = ['정치', '경제', '사회', 'IT', '과학', '자동차', '부동산',
            '생활', '세계', '의학', '인테리어', '예술', '연예', '스포츠']

filename='namu_test_article.txt'
filename4 = './hub/namu.model'


def make_DF(filename):
    file = codecs.open(filename, 'r', encoding='utf-8')
    text = file.read()

    twitter = Okt()
    word_dic = {}
    lines = text.split('\r\n')
    for line in lines:
        malist = twitter.pos(line)
        for taeso, pumsa in malist:
            if pumsa == 'Noun':
                if not (taeso in word_dic):
                    word_dic[taeso] = 0
                word_dic[taeso] += 1
    print(word_dic)
    keys = sorted(word_dic.items(), key=lambda x: x[1], reverse=True)

    top20_dic = {}
    if len(keys)>20:
        for word, count in keys[:20]:
            top20_dic[word]=count
    else:
        for word, count in keys:
            top20_dic[word]=count
    return top20_dic


print('Model test')
t1 = time.time()
model = word2vec.Word2Vec.load(filename4)
t2 = time.time()
print('model load elapsed=', t2-t1)
top20_dic=make_DF(filename)
for ks in top20_dic.keys():
    print(ks, top20_dic[ks], end=" ,")

# 카테고리별 단어의 유사도
cascores=[]
for ca in category:
    sims = []
    dfs = []
    for ks in top20_dic.keys():
        try:
            v1 = model.similarity(ca, ks)
            sims.append( v1 )
        except KeyError:
            sims.append( 0.0 )
        v2 = top20_dic[ks]
        dfs.append( v2 )
        print(ca, ks, 'similarity=',v1, 'df=',v2)

    sims = np.asarray(sims)
    dfs = np.asarray(dfs)

    # 단어출연 빈도를 가중치로 한 스코어
    val = np.dot(sims, dfs)
    print('wsum=', val)
    sco=val/ np.sum(dfs)
    print('scor=', sco)
    cascores.append(sco)

cascores=np.asarray(cascores)
maxidx = np.argmax(cascores)

# print(category)
# print(cascores, maxidx)

categorydic = {
    cate:scor for cate, scor in zip(category, cascores)
}
pc=sorted(categorydic, key=lambda k:categorydic[k], reverse=True)
print(pc)
print( sorted(cascores, reverse=True) )
print( 'predict=',pc[0],"/",pc[1] )

