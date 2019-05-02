from gensim.models import word2vec

model=word2vec.Word2Vec.load('toji.models')

print ( model.most_similar(positive=["땅"]) )


