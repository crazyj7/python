'''
knowldge management system.
create ; train dictionary.
'''
import codecs
from bs4 import BeautifulSoup
from konlpy.tag import Twitter
from gensim.models import word2vec

# config....
datafile_enc="utf-8"
datafile = "toji01.txt"             # text file book. (incuding puctuation, josa, eurmi)
datafile_wakati = "toji01.wakati"     # just word (noun, verb)...
datafile_model = 'toji.model'       # work2vec model

# Parsing XML and make wakati.
if False:
    print('Parsing XML')
    fp = codecs.open(datafile, "r", encoding=datafile_enc)
    soup = BeautifulSoup(fp, "html.parser")
    body = soup.select_one("text > body")
    text = body.getText()

    # 텍스트를 한 줄씩 처리하기 --- (※2)
    twitter = Twitter()
    results = []
    lines = text.split("\r\n")

    for line in lines:
        # 형태소 분석하기 --- (※3)
        # 단어의 기본형 사용
        malist = twitter.pos(line, norm=True, stem=True)
        r = []
        for word in malist:
            # 어미/조사/구두점 등은 대상에서 제외
            if not word[1] in ["Josa", "Eomi", "Punctuation"]:
                r.append(word[0])
        rl = (" ".join(r)).strip()
        results.append(rl)
        print(rl)

    # 파일로 출력하기  --- (※4)
    with open(datafile_wakati, 'w', encoding='utf-8') as fp:
        fp.write("\n".join(results))


# Word2Vec 모델 만들기 ---
print('making word2vec model....')
data = word2vec.LineSentence(datafile_wakati)
model = word2vec.Word2Vec(data, 
    size=200, window=10, hs=1, min_count=2, sg=1)
model.save(datafile_model)
print("ok")


# test model
model1 = word2vec.Word2Vec.load(datafile_model)
print('similar 땅=', model1.most_similar(positive=["땅"]))


