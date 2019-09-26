#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# 경고 무시 (쓸데없는 로그 출력금지)
import warnings
warnings.filterwarnings('ignore')

# 그래프 관련 
import matplotlib.pyplot as plt
import seaborn as sns



# 학습
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import random

import keras
from keras import regularizers
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, MaxPool2D, Flatten, Dense, Softmax
from keras.preprocessing import image

from keras.utils import np_utils
from keras.models import load_model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, K
from keras.models import Input, Model
from sklearn.model_selection import StratifiedKFold

from keras.preprocessing.image import ImageDataGenerator
import gc

########################
# Seed All
# import numpy as np
import tensorflow as tf
import random
# import os
# from keras import backend as K
import warnings
from keras.callbacks import Callback
from datetime import datetime
from pytz import timezone, utc


# In[ ]:


## load data
inputdir='../input/Kannada-MNIST/'
outputdir='./'
# datadir='../input/mykannada/'
datadir='/kaggle/input/mykannada/'


dftrain = pd.read_csv(inputdir+'train.csv')
dftest = pd.read_csv(inputdir+'test.csv')
dfsub = pd.read_csv(inputdir+'sample_submission.csv')


# In[ ]:


dftrain.head()


# In[ ]:


dftest.head()


# In[ ]:


nptrain = np.asarray(dftrain.iloc[:,1:].values)
print('test shape=', nptrain.shape)

nptest = np.asarray(dftest.iloc[:,1:].values)
print('test shape=', nptest.shape)


# In[ ]:


nptrain=nptrain.reshape((-1,28,28,1))
print('train shape=', nptrain.shape)
nptest=nptest.reshape((-1,28,28,1))
print('test shape=', nptest.shape)


# In[ ]:


def image_show(npdata, labels, cnt, brandom=True):
    plt.figure(figsize=(6*cnt, 6))
    if brandom:
        idx=np.random.randint(0,npdata.shape[0], cnt)
    else:
        idx=np.arange(cnt)
    for i in range(cnt):
        plt.subplot(1,cnt, i+1)
        plt.title(labels[idx[i]])
        imgdata = npdata[idx[i]].squeeze()
        plt.imshow(imgdata, cmap='gray')
    plt.show()


# In[ ]:


# train 이미지 랜덤 보기
if False:
    image_show(nptrain, dftrain.label, 6)


# In[ ]:


# test 이미지 랜덤 보기
if False:
    image_show(nptest, dftest.id, 6)


# In[ ]:


dftrain.label.value_counts()


# ## Image Augument

# In[ ]:


datagen1 = ImageDataGenerator(rescale=1./255, shear_range=0.1, zoom_range=0.1, 
                              horizontal_flip=False, vertical_flip=False,
                              width_shift_range=0.1, height_shift_range=0.1, 
                              rotation_range=15, brightness_range=[0.5, 1.2],
                              fill_mode='nearest')
datagen2 = ImageDataGenerator(rescale=1./255)

if False:
    datagen1.fit(nptrain)
    for xbatch, ybatch in datagen1.flow(nptrain, dftrain['label'], batch_size=6):
        image_show(xbatch, ybatch, 6, False)
        break


# In[ ]:

if False:
    for xbatch, ybatch in datagen1.flow(nptrain, dftrain['label'], batch_size=6):
        image_show(xbatch, ybatch, 6, False)
        break


# ## Train
# 

# In[ ]:


KST = timezone('Asia/Seoul')

def dbgprint(msg):
    os.system(f'echo \"{msg}\"')
    print(msg) 

class EpochLogWrite(Callback):
    def on_epoch_begin(self, epoch, logs={}):
        tmx = utc.localize(datetime.utcnow()).astimezone(KST).time()
        dbgprint('Epoch #{} begins at {}'.format(epoch+1, tmx))
    def on_epoch_end(self, epoch, logs={}):
        tmx = utc.localize(datetime.utcnow()).astimezone(KST).time()
        dbgprint('Epoch #{} ends at {}  acc={} val_acc={} '.format(epoch+1, tmx, round(logs['acc'],4), round(logs['val_acc'],4)))


SEED=1234
dbgprint('hello world. SEED={}'.format(SEED))

def seed_All():
    np.random.seed(SEED)
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    config = tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
    tf.set_random_seed(SEED)
    sess = tf.Session(graph=tf.get_default_graph(), config=config)
    K.set_session(sess)

seed_All()


# In[ ]:


###############################################
# configure
bDebug=False

# running mode
# 0: train.
# 1: test mode n.
# 2: ensemble use all models.
runmode = 2

epochs = 20


################################################
# !!! curFold 1~6 make 6 models...
curFold = 6  # make current fold (1..fold_k)

batch_size=32  # 16, 32, 64  debug.. memory dependent!
# K fold
fold_k = 6
imagesize=28


# In[ ]:


if bDebug:
    print('Debug Version!!!! ')
    print('Fast Fast. Only use 1000 items in train.')
    dftrain = dftrain[:1000]
    nptrain = nptrain[:1000]
    epochs = 10

# In[ ]:

method = 'conv'
modelname = 'kannada-v1-'
modellist = [modelname+'1-', modelname+'2-', modelname+'3-', 
             modelname+'4-', modelname+'5-', modelname+'6-']


# In[ ]:
if runmode==1:
    # test model number? 1~6
    files = glob.glob(datadir+modelname+'1-*')
    if len(files)>0:
        mp = max(files, key=os.path.getctime)
        modelpath = mp
    print('modelpath=', modelpath)


# In[ ]:


skf = StratifiedKFold(fold_k, random_state=SEED)


# In[ ]:


if runmode==0:
    # train
   
    kk=0
    for modelname, (tri, tei) in zip(modellist, skf.split(nptrain, dftrain['label'])):
        kk+=1
        # 아래를 주석처리하면 전체 모델 생성으로 오랜 시간 소요.
        # curFold (1~6) 해당 모델 1개만 생성.
        if kk != curFold:
            continue
        dbgprint('Make Model={}'.format(kk))

        modelpath = outputdir+modelname+method+'-{epoch:03d}-{val_acc:.4f}.ckpt'
        print(modelpath)

        nptrain_t = nptrain[tri,:]
        nptrain_v = nptrain[tei,:]

        dflabel_t = np_utils.to_categorical(dftrain.loc[tri]['label'])
        dflabel_v = np_utils.to_categorical(dftrain.loc[tei]['label'])

        datagen1 = ImageDataGenerator(rescale=1./255, shear_range=0.1, zoom_range=0.1, 
                                      horizontal_flip=False, vertical_flip=False,
                                      width_shift_range=0.1, height_shift_range=0.1, 
                                      rotation_range=15, brightness_range=[0.5, 1.2],
                                      fill_mode='nearest')
        datagen2 = ImageDataGenerator(rescale=1./255)

        datagen1.fit(nptrain_t)

        train_generator = datagen1.flow(nptrain_t, dflabel_t, batch_size=batch_size, seed=SEED, shuffle=True)
        val_generator = datagen1.flow(nptrain_v, dflabel_v, batch_size=batch_size,shuffle=False)

        # print(train_generator.class_indices)

        ### checkpoint save weights in progress...
        cp_callback = ModelCheckpoint(modelpath,  monitor='val_acc', mode='max', save_best_only=True,
                                      save_weights_only=False)
        es_callback = EarlyStopping(monitor='val_acc',  mode='max', patience=10)

        # tensorboard log
        if not os.path.exists('log'):
            os.mkdir('log')
        tensorboard = TensorBoard(log_dir='log/'+str(time.time()))

        model = Sequential()

        model.add(Conv2D(filters = 32, kernel_size = (4,4),padding = 'Same', activation ='relu', 
                         input_shape = (28,28,1)))
        model.add(Conv2D(filters = 32, kernel_size = (4,4),padding = 'Same', activation ='relu'))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(filters = 64, kernel_size = (4,4),padding = 'Same', activation ='relu'))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(filters = 128, kernel_size = (4,4),padding = 'Same', activation ='relu'))
        model.add(Dropout(0.25))

        model.add(Conv2D(filters = 256, kernel_size = (4,4),padding = 'Same', activation ='relu'))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(filters = 128, kernel_size = (4,4),padding = 'Same', activation ='relu'))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(10, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

        # model.summary()

    #     cblist = [tensorboard, cp_callback, es_callback, EpochLogWrite()]
        cblist = [EpochLogWrite(), es_callback, cp_callback]
        hist = model.fit_generator( train_generator, initial_epoch=0, epochs = epochs, validation_data=val_generator,
                                   callbacks=cblist, steps_per_epoch=len(tri)/batch_size, validation_steps=len(tei)/batch_size)

        model.save(outputdir+'model.h5')   # create model file in CWD
elif runmode==1:
    # test
    print('test : load model')
    model = load_model(modelpath) # load model from modelpath (dataset)
    # model.summary()


# In[ ]:

if runmode==1:
    # test mode
    datagen1 = ImageDataGenerator(rescale=1./255)
    dflabel = np_utils.to_categorical(dftrain['label'])
    print(nptrain.shape)
    print(dflabel.shape)
    eval_generator = datagen1.flow(nptrain, dflabel, batch_size=batch_size, shuffle=False)
    score = model.evaluate_generator(eval_generator, steps=dflabel.shape[0]/batch_size)
    # loss and acc
    print(score)

    # predict
    eval_generator.reset()
    result = model.predict_generator(eval_generator, steps=dflabel.shape[0]/batch_size)
    
    predict_result = np.argmax(result, axis=1)
    print(predict_result)
    print(dftrain['label'].values)
    print('match cnt=', np.sum(predict_result==dftrain['label'].values))    


# In[ ]:


if runmode==1:
    # test mode
    datagen1 = ImageDataGenerator(rescale=1./255)
    test_generator = datagen1.flow(nptest, batch_size=batch_size, shuffle=False)
    result = model.predict_generator(test_generator, steps=len(nptest)/batch_size)
    predict_result = np.argmax(result, axis=1)
    print(predict_result)

    dfsub['label'] = predict_result
    dfsub.to_csv('submit.csv', index=False)

# In[ ]:


if runmode==2:
    print("ensemble model.")
    datagen1 = ImageDataGenerator(rescale=1. / 255)
    test_generator = datagen1.flow(nptest, batch_size=batch_size, shuffle=False)

    files = glob.glob(datadir+modelname+'*')
    resultlist=[]
    for modelpath in files:
        print('modelpath=', modelpath)
        model = load_model(modelpath)

        # check model result
        test_generator.reset()
        result = model.predict_generator(test_generator, steps=len(nptest)/batch_size)
        resultlist.append(result)

    resultlist = np.asarray(resultlist)
    dbgprint(resultlist)
    dbgprint(resultlist.shape)

    prk = np.mean(resultlist, axis=0)
    dbgprint(prk)
    dbgprint(prk.shape)

    predict_result = np.argmax(prk, axis=1)
    dbgprint(predict_result)

    dfsub['label'] = predict_result
    dfsub.to_csv('submit.csv', index=False)

