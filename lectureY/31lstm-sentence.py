import codecs
from bs4 import BeautifulSoup
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.models import load_model

from keras.utils.data_utils import get_file
import numpy as np
import random, sys, os


modelname = '31lstm.model'


if False:
    fp = codecs.open("./aaa.txt", "r", encoding="utf-8")
    soup = BeautifulSoup(fp, "html.parser")
    body = soup.select_one("body")
    text = body.getText()+ " "
    print('length = ', len(text))

with open('toji01.txt', encoding='utf-8') as fp:
    text = fp.read()

# letter, id make.
chars = sorted(list(set(text)))
print('letter count=', len(chars))
char_indices = dict((c, i) for i,c in enumerate(chars))
indices_char = dict((i, c) for i,c in enumerate(chars))

# 텍스트를 maxlen으로  자르고, 다음에 오는 문자를 등록 (x,y)
maxlen = 20
step = 3
sentences=[]
next_char=[]
for i in range(0, len(text)-maxlen, step):
    sentences.append(text[i:i+maxlen])
    next_char.append(text[i+maxlen])

# 20자를 보고 다음문자가 어떤게 나오는지 조사하여 학습.
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
Y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, c in enumerate(sentence):
        X[i, t, char_indices[c]] = 1
    Y[i, char_indices[next_char[i]]] = 1

# Build Model
if not os.path.exists(modelname):
    model = Sequential()
    model.add( LSTM(128, input_shape=(maxlen, len(chars))))
    model.add( Dense(len(chars)))
    model.add( Activation('softmax'))
    optimizer = RMSprop(lr=0.01)
    model.compile(optimizer, loss='categorical_crossentropy')
    model.save('31lstm.model')
else:
    model = load_model(modelname)

# sample select
def sample(preds, tp=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / tp
    exp_preds = np.exp(preds)
    preds = exp_preds  / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# create text.
for iteration in range(1,10):
    print()
    print('-'*50)
    print('inter=', iteration)
    model.fit(X, Y, batch_size=128, nb_epoch=1)
    model.save('31lstm.model')

    start_index = random.randint(0, len(text)-maxlen-1)
    for diver in [0.2, 0.5, 1.0, 1.2]:
        print('diver=', diver)
        generated=''
        sentence = text[start_index:start_index+maxlen]
        generated += sentence
        print('--seed=', sentence)

        # auto create by seed.
        for i in range(400):
            x = np.zeros((1, maxlen, len(chars)))
            for t, c in enumerate(sentence):
                x[0, t, char_indices[c]]=1
            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diver)
            next_char = indices_char[next_index]
            generated+=next_char
            sentence = sentence[1:]+next_char

        print(generated)

