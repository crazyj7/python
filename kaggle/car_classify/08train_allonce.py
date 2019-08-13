import sys
import pandas as pd

from keras.applications import xception
from keras.applications import resnet50
from keras.applications import mobilenetv2
from efficientnet import EfficientNetB3

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

from keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
import gc


cache=False

inputdir = './'
outputdir = './'

imgwidth=224
imgheight=224

batch_size=32  # 16, 32, 64  debug

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


# xception, resnet50, mobilenetv2, efficientnetb3
method = 'xception'
modelname = 'carmodel-v8-'
# 'carmodel-v8-1-', 'carmodel-v8-6-', 'carmodel-v8-7-', 'carmodel-v8-8-',            
for fold_c, modelname in enumerate(['carmodel-v8-1-', 'carmodel-v8-6-', 'carmodel-v8-7-', 'carmodel-v8-8-','carmodel-v8-9-', 'carmodel-v8-10-']):
    files = glob.glob('./'+modelname+'*')
    mp = max(files, key=os.path.getctime)
    print('model=', mp)
        
    method='xception'
    if mp.find('resnet50')>0:
        method = 'resnet50'
    elif mp.find('mobilenetv2')>0:
        method = 'mobilenetv2'
    elif mp.find('efficientnetb3')>0:
        method = 'efficientnetb3'
    
    print('start learning...')
    print(fold_c, method)
    
    x_trainall = np.load(outputdir+'x_train.npy')
    y_trainall = np.load(outputdir+'y_train.npy')
    dfclass = pd.read_csv(inputdir+'class.csv')
    
    # append pseudo train
    if True:
        print('append pseudo label data!!!!')
        x_pseudo = np.load('x_pseudo.npy')
        y_pseudo = np.load('y_pseudo.npy')
        x_trainall = np.concatenate([x_trainall, x_pseudo])
        y_trainall = np.concatenate([y_trainall, y_pseudo])
    
    modelpath = modelname+method+'-{epoch:03d}-{new_score:.4f}.ckpt'
    print('modelpath=', modelpath)
    print('new train(pseudo include) size=', x_trainall.shape, y_trainall.shape)
    print(np.min(y_trainall), np.max(y_trainall))

    y_trainall_onehot = np_utils.to_categorical(y_trainall, 196)

    # Image Augumentation
    datagen1 = ImageDataGenerator(rescale=1./255, shear_range=0.1, zoom_range=0.1, horizontal_flip=True, vertical_flip=False,
                                  width_shift_range=0.1, height_shift_range=0.1,
                                  fill_mode='nearest', preprocessing_function = get_random_eraser(v_l=0, v_h=1),)
    train_generator = datagen1.flow(x_trainall, y_trainall_onehot, batch_size=batch_size)

    ### checkpoint save weights in progress...
    cp_callback = ModelCheckpoint(modelpath,  monitor='new_score', mode='max', save_best_only=True, save_weights_only=True)
    es_callback = EarlyStopping(monitor='new_score',  mode='max', patience=10, min_delta=0.0001)

    # tensorboard log
    if not os.path.exists('log'):
        os.mkdir('log')
    tensorboard = TensorBoard(log_dir='log/'+str(time.time()))

    # In[10]:
    inputs = Input(shape=(224,224,3))
    net = None
    print('method=', method)
    if method=='xception':
        net = xception.Xception(input_tensor=inputs, input_shape=(224, 224, 3), include_top=False, weights='imagenet', pooling='max')
    elif method=='resnet50':
        net = resnet50.ResNet50(input_tensor=inputs, input_shape=(224, 224, 3), include_top=False, weights='imagenet', pooling='max')
    elif method=='mobilenetv2':
        net = mobilenetv2.MobileNetV2(input_tensor=inputs, input_shape=(224, 224, 3), include_top=False, weights='imagenet', pooling='max')
    elif method=='efficientnetb3':
        net = EfficientNetB3(input_tensor=inputs, input_shape=(224, 224, 3), include_top=False, weights='imagenet', pooling='max')
 
    net2 = Dense(224, activation='relu') (net.layers[-1].output)
    net2 = Dense(196)(net2)
    net2 = Softmax(196)(net2)
    model = Model(inputs=inputs, outputs=net2)
    # model.summary()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', new_score])

    # continuous learning
    print('load_weights :', mp)
    model.load_weights(mp)
    
    epochs=10
    hist = model.fit_generator( train_generator, initial_epoch=0, epochs = epochs,  
                               callbacks=[tensorboard, cp_callback, es_callback],
                               steps_per_epoch=len(x_trainall)/batch_size)
