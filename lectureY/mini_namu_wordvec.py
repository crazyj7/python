import ijson
import codecs
import sys, os
import time

import nltk

from gensim.models import word2vec
from konlpy.tag import Okt

import pickle


'''
JSON 나무위키 파싱

'''


def load_json(filename):
    count=0
    with open(filename, 'r') as fd:
        # parser = ijson.parse(fd)
        # for prefix, event, value in parser:
        #      print ( 'prefix=', prefix, 'event=', event )

        #     if prefix.endswith('.title'):
        #         # print('index=', count+1)
        #         # print("\nTITLE: %s" % value)
        #     elif prefix.endswith('.text'):
        #         # print("\nCONTENT: %s" % value)
        #         count += 1
        # print ('total cnt=', count)
        for item in ijson.items(fd, 'item'):
            print(item['title'])
            # print(item['text'])

def load_and_write_content(filename, filename2):
    count=0
    file = codecs.open(filename2, 'w', encoding='utf-8')
    with open(filename, 'r') as fd:
        for item in ijson.items(fd, 'item'):
            count+=1
            file.write('[[제목]]: ')
            file.write(item['title'])
            file.write('\n')
            file.write('[[내용]]: \n')
            file.write(item['text'])
            file.write("\n")
    file.close()
    print('contents count=', count)


def make_wakati(f1, f2, f3):
    file = codecs.open(f1, 'r', encoding='utf-8')
    text = file.read()

    twitter = Okt()
    lines = text.split('\r\n')
    results = []
    for line in lines:
        r = []
        malist = twitter.pos(line)      # part of speech (POS)
        # pumsa : Noun, Josa, Verb, Eomi, Punctuation, Number, KoreanParticle, 
        for word, pumsa in malist:
            if not pumsa in ['Josa', 'Eomi', 'Punctuation']:
                r.append(word)
        results.append((" ".join(r)).strip())
    output = (" ".join(results)).strip()

    with open(f2, "w", encoding="utf-8") as fp:
        fp.write(output)

    data = word2vec.LineSentence(f2)
    model = word2vec.Word2Vec(data, size=200, window=10, hs=1, min_count=5, sg=1, workers=3)
    model.save(f3)


if __name__ == "__main__":
    print('load_json start')
    print('curdir=', os.getcwd())
    filedir = os.path.dirname( os.path.realpath(__file__))
    os.chdir(filedir)
    print('filedir=', filedir)

    # filename ='/home/psychic/download/namuwiki_20180326.json'
    filename ='mini_namu.json'
    filename2='mini_namu.txt'
    filename3='mini_namu.wakati'
    filename4='mini_namu.model'
    filename5='mini_namu.pkl'

    # load_json(filename)
    # 나무위키 JSON DB에서 제목과 컨텐트를 스트링으로 기록한 파일 생성. (txt)
    if False:
        print('Create WordTxT')
        load_and_write_content(filename, filename2)
        print('End WordTxt ')

    if False:
        print('Create Wakati')
        t1=time.time()
        make_wakati(filename2, filename3, filename4)
        t2=time.time()
        print('End Wakati', 'time=',t2-t1)


    if False:
        print('Model test')
        t1=time.time()
        model = word2vec.Word2Vec.load(filename4)
        t2=time.time()
        print(model.most_similar(positive=["조선", "일본"]))
        print(model.most_similar(positive=["고려"]))
        print(model.most_similar(positive=["고려"], negative=["왕"]))
        print(model.most_similar(positive=["고려", "경제"]))
        print(model.most_similar(positive=["RPG", "게임"]))
        print(model.most_similar(positive=["임진왜란"]))
        print(model.most_similar(positive=["왕", "여자"], negative=["남자"]))

        print(model.similarity('고려','공민왕'))
        print(model.doesnt_match("아침 점심 저녁 조선 밥".split() ) )
        print(model.doesnt_match("총 무기 칼 게임 하늘".split() ) )

        print('time=',t2-t1)


    # NLTK test
    if True:

        t1 = time.time()
        if True:
            print('make tokens nltk')
            t = Okt()
            doc_ko = open(filename2, 'r').read()
            tokens = t.morphs(doc_ko)
            print(tokens)

            ko = nltk.Text(tokens, name='none')
            print(len(ko.tokens))
            print(len(set(ko.tokens)))
            ko.vocab()

            with open(filename5, 'wb') as fp:
                pickle.dump(ko, fp)
        else:
            with open(filename5, 'rb') as fp:
                ko = pickle.load(fp)

        t2=time.time()
        print('time=', t2-t1)

        print('count word=조선', ko.count('조선'))
        print('similar word=조선', ko.similar('조선'))
        print('line word=조선', ko.concordance('조선'))
        print('line word=조선', ko.concordance('조선'))

        ko.

        ko.plot(50)


    print('end')



