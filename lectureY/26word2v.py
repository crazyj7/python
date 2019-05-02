
from gensim.models import word2vec
import codecs
from konlpy.tag import Okt

file = codecs.open('toji01.txt', 'r', encoding='utf-8')
text = file.read()

twitter = Okt()
lines = text.split('\r\n')
results=[]
for line in lines:
    r=[]
    malist = twitter.pos(line)
    for word, pumsa in malist:
        if not pumsa in ['Josa', 'Eomi', 'Punctuation']:
            r.append(word)
    results.append((" ".join(r)).strip())
output=(" ".join(results)).strip()
print(output)

with open("toji01.wakati", "w", encoding="utf-8") as fp:
    fp.write(output)

data = word2vec.LineSentence("toji01.wakati")
model = word2vec.Word2Vec(data, size=200, window=10, hs=1, min_count=2, sg=1)
model.save('toji.models')


