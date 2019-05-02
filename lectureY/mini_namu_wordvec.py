import ijson
import codecs
import sys, os
import time

import nltk

from gensim.models import word2vec
from konlpy.tag import Okt

import pickle
from textwrap import wrap

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
    file = codecs.open(filename2, 'w', encoding='utf-8', errors='ignore')
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
    file.close()

    # make wakati file
    twitter = Okt(max_heap_size=1024*8)     # heap size.. 8G
    lines = text.split('\r\n')
    t1=time.time()

    print('making wakati start')
    print('lines count=', len(lines))
    fp = open(f2, "w", encoding="utf-8")
    for line in lines:
        linelen = len(line)
        print("line length=", linelen)

        # split.
        # lineparts = map(''.join, zip(*[iter(line)] * 1000*1000*20))    # 20MB / 1line
        blocksize = 1000*1000*20
        if linelen % blocksize == 0:
            blockcnt = int(linelen/blocksize)
        else:
            blockcnt = int(linelen / blocksize) + 1

        for li in range(blockcnt):
            if li==blockcnt-1:
                linepart = line[li*blocksize:]
            else:
                linepart = line[li*blocksize:(li+1)*blocksize]
            print('progress=', li, '/', blockcnt, len(linepart) )
            malist = twitter.pos(linepart)      # part of speech (POS)
            # pumsa : Noun, Josa, Verb, Eomi, Punctuation, Number, KoreanParticle,
            for word, pumsa in malist:
                if not pumsa in ['Josa', 'Eomi', 'Punctuation']:
                    fp.write(word.strip())
                    fp.write(" ")
    fp.close()
    t2=time.time()
    print('making wakati end time=', t2-t1)

    # make word2vec
    t1=time.time()
    print('word2vec start.')
    data = word2vec.LineSentence(f2)
    model = word2vec.Word2Vec(data, size=200, window=10, hs=1, min_count=5, sg=1, workers=3)
    model.save(f3)
    t2=time.time()
    print('word2vec end. time=', t2-t1)


if __name__ == "__main__":
    print('load_json start')
    print('curdir=', os.getcwd())
    filedir = os.path.dirname( os.path.realpath(__file__))
    os.chdir(filedir)
    print('filedir=', filedir)

    # mini namu
    filename ='mini_namu.json'
    filename2='mini_namu.txt'
    filename3='mini_namu.wakati'
    filename4='mini_namu.models'
    filename5='mini_namu.pkl'

    #real namu
    filename ='./hub/namu.json'
    filename2='./hub/namu.txt'
    filename3='./hub/namu.wakati'
    filename4='./hub/namu.models'
    filename5='./hub/namu.pkl'

    # load_json(filename)
    # 나무위키 JSON DB에서 제목과 컨텐트를 스트링으로 기록한 파일 생성. (txt)
    if False:
        print('Create WordTxT')
        t1=time.time()
        load_and_write_content(filename, filename2)
        t2=time.time()
        print('End WordTxt ', 'time=',t2-t1)

    if False:
        print('Create Wakati')
        t1=time.time()
        make_wakati(filename2, filename3, filename4)
        t2=time.time()
        print('End Wakati', 'time=',t2-t1)

    if True:
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
    if False:

        t1 = time.time()
        if True:
            print('make tokens nltk')
            t = Okt()
            doc_ko = open(filename3, 'r', encoding='utf-8').read()

            # tokens = t.morphs(doc_ko)
            # tokens = t.nouns(doc_ko)
            # print(tokens)

            tokens = []
            malist = t.pos(doc_ko)
            for word, pumsa in malist:
                # pumsa list ; Adjective, Adverb, Alpha, Conjunction, Determiner
                # Eomi, Exclamation, Foreign, Hashtag, Josa, KoreanParticle, Noun, Number
                # PreEomi, Punctuation, ScreenName, Suffix, Unknown, Verb.
                if not pumsa in ['Josa', 'Eomi', 'Punctuation', 'Unknown', 'Conjunction', 'Suffix']:
                    tokens.append(word)

            # NLTK TEXT 객체 생성. 개수, 유사성, 검색.
            ko = nltk.Text(tokens, name='none')
            print(len(ko.tokens))
            print(len(set(ko.tokens)))
            ko.vocab()

            # NLTK FreqDist 객체 생성. 개수, 빈도측정. 상위 발생 단어.
            fd = nltk.FreqDist(tokens)

            with open(filename5, 'wb') as fp:
                pickle.dump(ko, fp)
                pickle.dump(fd, fp)
        else:
            with open(filename5, 'rb') as fp:
                ko = pickle.load(fp)
                fd = pickle.load(fp)

        t2=time.time()
        print('time=', t2-t1)

        # Text object methods.
        # print('count word=조선', ko.count('조선'))
        # print('similar word=고려')
        # ko.similar('고려')    # auto print
        # print('line word=조선')
        # ko.concordance('조선')    # auto print
        # ko.plot(50)

        # FreqDist Object methods.
        # fd = nltk.FreqDist(["조선", "고려", "신라", "조선"])
        print( fd.N(), fd["조선"], fd.freq("조선"))
        print( fd.most_common(10) )




    print('end')



