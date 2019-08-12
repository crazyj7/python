#!/usr/bin/env python
# coding: utf-8

# # Train car - v7

# ```
# data ; crop and resize ; add margin!
# network ; xception 
# 
# ImageAugment  ; shift add. 
# # random_eraser ; image part remove ; off
# 
# batch 32
# earlystop ; f1_score 
# checkpoint best save 
# 
# train acc;  , val acc;  , lb acc;   
# model1 ;  
# model2;  
# model3 ; 
# 
# enssamble. sum.
# 
# 
# ```
# 

# ## load package

# In[1]:


import sys
import pandas as pd

from keras.applications import xception
from keras.preprocessing import image
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import keras
from keras import regularizers
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense, Softmax
import random
from keras.utils import np_utils
from keras.models import load_model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, K
from keras.models import Input, Model
from sklearn.metrics import f1_score

from keras.preprocessing.image import ImageDataGenerator
import seaborn as sns


# In[2]:


cache=False
basedir = './'
imgwidth=224
imgheight=224
# get_ipython().run_line_magic('matplotlib', 'inline')

# K fold
fold_k = 5
# current fold
fold_c = 5   # 1~fold_k


for fold_c in range(1, 6):
    # ## TRAIN

    # In[3]:


    x_trainall = np.load('x_train.npy')
    y_trainall = np.load('y_train.npy')
    dfclass = pd.read_csv(basedir+'class.csv')


    # In[4]:


    # cross validation
    datacnt = x_trainall.shape[0]
    flagval = np.zeros(datacnt)
    modelpath = 'car-v7-'+str(fold_c)+'.ckpt'

    print('modelpath=', modelpath)
    flagval[(fold_c-1)*2000:(fold_c)*2000] = 1

    x_train = x_trainall[flagval==0]
    y_train = y_trainall[flagval==0]
    x_val = x_trainall[flagval==1]
    y_val = y_trainall[flagval==1]

    del x_trainall
    del y_trainall
    print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
    print(np.min(y_train), np.max(y_train))


    # In[5]:


    y_train_onehot = np_utils.to_categorical(y_train, 196)
    y_val_onehot = np_utils.to_categorical(y_val, 196)


    # In[6]:


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


    # In[7]:


    # Image Augumentation
    batch_size=32  # 32, 64
    datagen1 = ImageDataGenerator(rescale=1./255, shear_range=0.1, zoom_range=0.1, horizontal_flip=True, vertical_flip=False,
                                  width_shift_range=0.1, height_shift_range=0.1,
                                  fill_mode='nearest', preprocessing_function = get_random_eraser(v_l=0, v_h=1),)
    # datagen1 = ImageDataGenerator(rescale=1./255, shear_range=0.1, zoom_range=0.1, horizontal_flip=True, vertical_flip=False,
    #                               width_shift_range=0.1, height_shift_range=0.1,
    #                               fill_mode='nearest')
    datagen2 = ImageDataGenerator(rescale=1./255)
    train_generator = datagen1.flow(x_train, y_train_onehot, batch_size=batch_size)
    val_generator = datagen2.flow(x_val, y_val_onehot, batch_size=batch_size, shuffle=False)


    # In[8]:


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


    # In[9]:


    ### checkpoint save weights in progress...
    cp_callback = ModelCheckpoint(modelpath,  monitor='val_new_score', mode='max', save_best_only=True, save_weights_only=True)
    es_callback = EarlyStopping(monitor='val_new_score',  mode='max', patience=20, min_delta=0.0001)

    # tensorboard log
    if not os.path.exists('log'):
        os.mkdir('log')
    tensorboard = TensorBoard(log_dir='log/'+str(time.time()))


    # In[10]:


    inputs = Input(shape=(224,224,3))
    net = xception.Xception(input_tensor=inputs, input_shape=(224, 224, 3), include_top=False, weights='imagenet', pooling='max')
    net2 = Dense(224, activation='relu') (net.layers[-1].output)
    net2 = Dense(196)(net2)
    net2 = Softmax(196)(net2)
    model = Model(inputs=inputs, outputs=net2)
    # model.summary()


    # In[11]:


    ##### model freeze. after acc 90.
    if False:
        print('freezing...')
        for layer in model.layers[:-4]:
            layer.trainable=False
    else:
        print('no freezing...')
    #     for layer in model.layers:
    #         print(layer, layer.trainable)
    #     model.summary()


    # In[12]:


    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', new_score])


    # In[ ]:


    if os.path.exists(modelpath):
        print('load weights...')
        model.load_weights(modelpath)
    else:
        print('not found weights... new')


    # In[13]:


    # epochs = 100
    hist = model.fit_generator( train_generator, initial_epoch=0, epochs = 500, validation_data=val_generator, 
                               callbacks=[tensorboard, cp_callback, es_callback],
                               steps_per_epoch=len(x_train)/batch_size, validation_steps=len(x_val)/batch_size)


# In[15]:


# hist = model.evaluate( x_val/255., y_val_onehot, batch_size=30, verbose=1 )
hist = model.evaluate_generator(val_generator, steps=len(x_val)/batch_size)
print(hist)


# In[16]:


model.save('car-v7.h5')


# ### Submission

# In[14]:


# test data load for submission
x_test = np.load('x_test.npy')
x_test = x_test/255.


# In[31]:


# one model submission 
if False:
    predictions = model.predict( x_test )
    pdi = np.argmax(predictions, axis=1)
    print(pdi, np.min(pdi), np.max(pdi))


# In[15]:


# ensamble. submission.
# model = load_mode('car-v7.h5')
if True:
#     mo = load_model(mp, custom_objects={'new_score': new_score})
    inputs = Input(shape=(224,224,3))
    net = xception.Xception(input_tensor=inputs, input_shape=(224, 224, 3), include_top=False, pooling='max')
    net2 = Dense(224, activation='relu') (net.layers[-1].output)
    net2 = Dense(196)(net2)
    net2 = Softmax(196)(net2)
    model = Model(inputs=inputs, outputs=net2)
    predictions=[]
    for ff in range(1, fold_k+1):
        mp='car-v7-'+str(ff)+'.ckpt'
        print('model',ff,':', mp)
        model.load_weights(mp)
        pr = model.predict( x_test )
        predictions.append(pr)
        print('prediction',ff,':',pr)
    predictions = np.asarray(predictions)
    prk = np.sum(predictions, axis=0 )
    pdi = np.argmax(prk, axis=1)
    print('final:', pdi, np.min(pdi), np.max(pdi))


# In[16]:


submission = pd.read_csv(basedir+'sample_submission.csv')
submission["class"] = pdi + 1  # class [0,195] to [1,196]  
submission.to_csv("submission.csv", index=False)
submission.head()


# In[17]:


sns.countplot(submission["class"], order=submission["class"].value_counts(ascending=True).index)


# In[ ]:





# In[ ]:





# In[ ]:




