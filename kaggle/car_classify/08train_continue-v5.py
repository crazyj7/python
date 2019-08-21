import sys
import pandas as pd

from keras.applications import xception
from keras.applications import resnet50
from keras.applications import mobilenetv2
# from efficientnet import EfficientNetB3

import time
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import random

import keras
from keras import regularizers
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense, Softmax
from keras.preprocessing import image

from keras.utils import np_utils
from keras.models import load_model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, K
from keras.models import Input, Model
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
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
KST = timezone('Asia/Seoul')

class EpochLogWrite(Callback):
    def on_epoch_begin(self, epoch, logs={}):
        tmx = utc.localize(datetime.utcnow()).astimezone(KST).time()
        dbgprint('Epoch #{} begins at {}'.format(epoch+1, tmx))
    def on_epoch_end(self, epoch, logs={}):
        tmx = utc.localize(datetime.utcnow()).astimezone(KST).time()
        dbgprint('Epoch #{} ends at {}  acc={} val_acc={} val_new_score={}'.format(epoch+1, tmx, round(logs['acc'],4), round(logs['val_acc'],4), round(logs['val_new_score'],4) ))


warnings.filterwarnings('ignore')

def dbgprint(msg):
	os.system(f'echo \"{msg}\"')
	print(msg)


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
#######################
# configure
bDebug=False
bKaggle = True
curFold = 6  # make current fold (1..fold_k)

batch_size=32  # 16, 32, 64  debug.. memory dependent!

# K fold
fold_k = 6

# datadir = './data_carmodel/'
inputdir='./'   # csv file
datadir='./pre/'   # preprocessing image
outputdir='./pre/'   # make model to
if bKaggle:
    datadir = '../input/carmodel5/ppcarmodel/'
    inputdir = '../input/2019-3rd-ml-month-with-kakr/'
    outputdir='./'

# imagesize=299
imagesize=250

#ref: https://github.com/yu4u/cutout-random-erasing/blob/master/cifar10_resnet.py
def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    def eraser(input_img):
        img_h, img_w, img_c = input_img.shape
        p_1 = np.random.rand()
        if p_1 > p:
            return input_img
        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)
            if left + w <= img_w and top + h <= img_h:
                break
        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.uniform(v_l, v_h)
        input_img[top:top + h, left:left + w, :] = c
        return input_img
    return eraser

# f1 score 
def new_score(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = (true_positives + K.epsilon()) / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = (true_positives + K.epsilon()) / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



dftrain = pd.read_csv(inputdir+'train.csv')
if bDebug:
    print('Debug Version!!!! ')
    dftrain = dftrain[:1000]
    
dftrain['class'] = dftrain['class'].astype('str')
dfclass = pd.read_csv(inputdir+'class.csv')
classes = list (str(num) for num in range(1,197))
print(classes)

# append pseudo train
if False:
    print('append pseudo label data!!!!')
    x_pseudo = np.load(datadir+'x_pseudo3.npy')
    y_pseudo = np.load(datadir+'y_pseudo3.npy')
    x_trainall = np.concatenate([x_trainall, x_pseudo])
    y_trainall = np.concatenate([y_trainall, y_pseudo])
        

# xception, resnet50, mobilenetv2, efficientnetb3
method = 'xception'
# method = 'resnet50'
modelname = 'carmodel-v5-'
# 'carmodel-v8-1-', 'carmodel-v8-6-', 'carmodel-v8-7-', 'carmodel-v8-8-', 'carmodel-v8-9-', 'carmodel-v8-10-'   
# , 'carmodel-v9-2-', 'carmodel-v9-3-', 'carmodel-v9-4-', 'carmodel-v9-5-'
modellist = ['carmodel-v5-1-', 'carmodel-v5-2-', 'carmodel-v5-3-', 'carmodel-v5-4-', 'carmodel-v5-5-', 'carmodel-v5-6-']

skf = StratifiedKFold(fold_k, random_state=SEED)
print(dftrain.head())

kk=0
for modelname, (tri, tei) in zip(modellist, skf.split(dftrain['img_file'], dftrain['class'])):
    kk+=1
    # 아래를 주석처리하면 전체 모델 생성으로 오랜 시간 소요.
    # curFold (1~6) 해당 모델 1개만 생성.
    if kk < curFold:
        continue
    if kk > curFold:
        break
        
    dbgprint('Make Model={}'.format(kk))
    
    print(tri, tei)
    print('train size=', len(tri), 'val size=', len(tei))
    files = glob.glob(datadir+modelname+'*')
    if len(files)>0:
        mp = max(files, key=os.path.getctime)
        print('model=', mp)
    else:
        print('model not exists')
        mp = datadir+modelname+'xception-0-0.ckpt'

    method='xception'
    if mp.find('resnet50')>0:
        method = 'resnet50'
    elif mp.find('mobilenetv2')>0:
        method = 'mobilenetv2'
    elif mp.find('efficientnetb3')>0:
        method = 'efficientnetb3'
    
    print('start learning...')
    print(kk, method)
    modelpath = outputdir+modelname+method+'-{epoch:03d}-{val_new_score:.4f}.ckpt'    
    
    dftrain_t = dftrain.iloc[tri,:].reset_index()
    dftrain_v = dftrain.iloc[tei,:].reset_index()

    # Image Augumentation
    datagen1 = ImageDataGenerator(rescale=1./255, shear_range=0.1, zoom_range=0.1, 
                                  horizontal_flip=True, vertical_flip=False,
                                  width_shift_range=0.1, height_shift_range=0.1, 
                                  rotation_range=15, brightness_range=[0.5, 1.2],
                                  fill_mode='nearest', preprocessing_function = get_random_eraser(v_l=0, v_h=255),)
    datagen2 = ImageDataGenerator(rescale=1./255)
    
    train_generator = datagen1.flow_from_dataframe( dataframe=dftrain_t, 
                                                   directory=datadir+'train', 
                                                   x_col = 'img_file', 
                                                   y_col = 'class', 
                                                   target_size = (imagesize, imagesize),
                                                   class_mode='categorical', 
                                                   classes = classes,
                                                   batch_size=batch_size, 
                                                  seed=SEED, 
                                                  shuffle=True)
    
    val_generator = datagen2.flow_from_dataframe( dataframe=dftrain_v, 
                                                   directory=datadir+'train', 
                                                   x_col = 'img_file', 
                                                   y_col = 'class', 
                                                   target_size = (imagesize, imagesize),
                                                   class_mode='categorical', 
                                                   classes = classes,
                                                   batch_size=batch_size, 
                                                  seed=SEED, 
                                                  shuffle=False)

    ### checkpoint save weights in progress...
    cp_callback = ModelCheckpoint(modelpath,  monitor='val_new_score', mode='max', save_best_only=True, save_weights_only=True)
    es_callback = EarlyStopping(monitor='val_new_score',  mode='max', patience=20)
    # , min_delta=0.0001

    # tensorboard log
    if not os.path.exists('log'):
        os.mkdir('log')
    tensorboard = TensorBoard(log_dir='log/'+str(time.time()))

    # In[10]:
    inputs = Input(shape=(imagesize,imagesize,3))
    net = None
    print('method=', method)
    if method=='xception':
        net = xception.Xception(input_tensor=inputs, input_shape=(imagesize, imagesize, 3), include_top=False, weights='imagenet', pooling='max')
    elif method=='resnet50':
        net = resnet50.ResNet50(input_tensor=inputs, input_shape=(imagesize, imagesize, 3), include_top=False, weights='imagenet', pooling='max')
    elif method=='mobilenetv2':
        net = mobilenetv2.MobileNetV2(input_tensor=inputs, input_shape=(imagesize, imagesize, 3), include_top=False, weights='imagenet', pooling='max')
    elif method=='efficientnetb3':
        net = EfficientNetB3(input_tensor=inputs, input_shape=(imagesize, imagesize, 3), include_top=False, weights='imagenet', pooling='max')
 
    net2 = Dense(256, activation='relu', kernel_initializer='he_normal') (net.layers[-1].output)
    net2 = Dropout(rate=0.4)(net2)
    net2 = Dense(196, activation='softmax', kernel_initializer='lecun_normal')(net2)
    model = Model(inputs=inputs, outputs=net2)
    # model.summary()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc', new_score])

    # continuous learning
    print('load_weights :', mp)
    if os.path.exists(mp):
        model.load_weights(mp)
        print('loaded model weights!!! continous train.')
    else:
        print('new start model.')
    
    # debug
# callbacks=[tensorboard, cp_callback, es_callback],
    epochs=200
    hist = model.fit_generator( train_generator, initial_epoch=0, epochs = epochs, validation_data=val_generator, 
                               callbacks=[tensorboard, cp_callback, es_callback, EpochLogWrite()],
                               steps_per_epoch=len(tri)/batch_size, validation_steps=len(tei)/batch_size)

