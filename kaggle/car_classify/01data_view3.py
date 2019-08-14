#!/usr/bin/env python
# coding: utf-8

# # Car Classify

# In[1]:
#
#
# image size 299
# cutout image
# margin 20px
#

import numpy as np
import keras
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
from PIL import Image
import seaborn as sns


# ## load data
# pandas dataframe
# dfclass, dftrain, dftest

# In[2]:


basedir = './'

# 클래스 데이터 읽기  
dfclass = pd.read_csv(basedir+'class.csv')


# In[3]:


dfclass.head()


# In[4]:


# 테스트 데이터 읽기  
dftest = pd.read_csv(basedir+'test.csv')
dftest.head()


# In[5]:


# 훈련 데이터 읽기  
dftrain = pd.read_csv(basedir+'train.csv')
dftrain.head()
dftrain['class']=dftrain['class']-1  # make class [0,195] from [1,196]
dftrain.describe()


# In[6]:


# shape 형상, 개수 확인   
dftrain.shape, dftest.shape, dfclass.shape


# In[7]:


print(dftrain.count(), dftest.count(), dfclass.count())


# In[8]:


# 빠진 파일 확인 
for f in dftest.img_file:
    if not os.path.exists(basedir+'test/'+f):
        print('not found:', f)


# In[9]:


for f in dftrain.img_file:
    if not os.path.exists(basedir+'train/'+f):
        print('not found:', f)


# In[10]:


# 클래스별로 개수를 보기. 균일하게 존재하는지 확인.
# 데이터 개수가 학습시 균일할수록 좋다. 한쪽에 너무 치우치면 학습 실패. 
# 마지막에 어떤 클래스의 데이터가 좀 많긴하다. 일단 그대로 학습 진행.. 
sns.countplot(dftrain["class"], order=dftrain["class"].value_counts(ascending=True).index)


# In[11]:


# 개수 확인하는 다른 방법. 오름 차순 정렬. 
# dftrain["class"].value_counts(ascending=True)


# ## image view

# In[12]:


# 이미지 파일 보기 
plt.figure()
pi=0
for i in range(3):
    a = dftrain.img_file[i]
    imgpath = basedir+'train/'+a
    img = plt.imread(imgpath)
    pi+=1
    plt.subplot(1,3,pi)
    plt.title(dftrain["class"][i])
    plt.imshow(img)
    print(img.shape)


# In[13]:


# 크기가 제각각인 이미지. 리사이즈.
# resize??? (224,224,3)

pi=0
for i in range(3):
    a = dftrain.img_file[i]
    imgpath = basedir+'train/'+a
    print(imgpath)
    pi+=1
    plt.subplot(1,3,pi)
    plt.title(dftrain["class"][i])
    plt.axis('off')
    imgobj = Image.open(imgpath)
    imgobj = imgobj.resize((224,224))
    img_r = np.asarray(imgobj)
    plt.imshow(img_r)
    print(img_r.shape)


# In[14]:


# 바인딩 박스 정보 확인. test
print(i, img.shape)
print(dftrain.loc[i])
bb = [dftrain.loc[i]['bbox_x1'], dftrain.loc[i]['bbox_y1'], dftrain.loc[i]['bbox_x2'], dftrain.loc[i]['bbox_y2']]
print(bb)


# ## bounding box

# In[15]:


# 이미지에 박스 그리기 
def bound(img, x1,y1,x2,y2):
    img[y1,x1:x2] = [255,0,0]
    img[y2,x1:x2] = [255,0,0]
    img[y1:y2,x1] = [255,0,0]
    img[y1:y2,x2] = [255,0,0]
    return img


# In[16]:


img2 = np.copy(img)
img2=bound(img2, bb[0], bb[1], bb[2], bb[3])
plt.imshow(img2)


# In[17]:


# 바인딩 박스가 이미지 리사이즈시 새로운 박스 위치를 찾기
# bb, width, height
def resize_bb(bb, w, h):
    rx = 224/w
    ry = 224/h
    return [int(bb[0]*rx), int(bb[1]*ry), int(bb[2]*rx), int(bb[3]*ry)]

bb2=resize_bb(bb, img.shape[1], img.shape[0])
print(bb, bb2)


# In[18]:


img2 = np.copy(img_r)
img2=bound(img2, bb2[0], bb2[1], bb2[2], bb2[3])
plt.imshow(img2)


# ## Training dataset preprocessing

# image (224,224,3) shape. 3 channels. crop box area.
# 

# In[27]:


# 이미지를 바인딩 박스만 추출하여 리사이즈 한다. (차만 보이게 추출하고 리사이즈)
''' crop and resize '''
def prepro_img(imgpath, bb=None, margin=20, size=(224,224)):
    img = Image.open(imgpath)
    img = img.convert('RGB')  # 3 channel image.
    imgn = np.asarray(img)

    #crop
    imgc = imgn
    if bb!=None:
        h, w, c = imgn.shape
        x1 = max(0, bb[0]-margin)
        y1 = max(0, bb[1]-margin)
        x2 = min(bb[2]+margin, w)
        y2 = min(bb[3]+margin, h)
    #     imgc = imgn[bb[1]:bb[3], bb[0]:bb[2]]
        imgc = imgn[y1:y2, x1:x2]
    
    newimg = imgc
    if size!=None:
        imgcobj = Image.fromarray(imgc)
        imgcobj2 = imgcobj.resize(size)
        newimg = np.asarray(imgcobj2)
    return newimg


# In[35]:


# preprocess test
i=100
imgpath = basedir+'train/'+dftrain.loc[i]['img_file']
bb = [dftrain.loc[i]['bbox_x1'], dftrain.loc[i]['bbox_y1'], dftrain.loc[i]['bbox_x2'], dftrain.loc[i]['bbox_y2']]
cl = dftrain.loc[i]['class']
img = prepro_img(imgpath, bb)
print(img.shape)

size=(400,400)
plt.figure(figsize=(8*3, 8))
plt.subplot(1,3,1)
plt.title('resize')
plt.imshow(prepro_img(imgpath, None, 0, size))

plt.subplot(1,3,2)
plt.title('crop')
plt.imshow(prepro_img(imgpath, bb, 0, size))

plt.subplot(1,3,3)
plt.title('cropmargin')
plt.imshow(prepro_img(imgpath, bb, 20, size))
plt.show()


# ## Make Train dataset

# In[36]:


# 훈련 데이터 만들기. 
#      주의! class값 범위는 0~195로 조정하여 저장. 실제 클래스값(1~196)을 알려면 +1 필요.
# crop and resize하여 저장
# make train data numpy 

force = True

cnt_train = dftrain.shape[0]
x_train=[]
y_train=[]

if not os.path.exists('x_train3.npy') or force :
    for i in range(cnt_train):
        if i%1000==0 :
            print(i, '/', cnt_train)
        imgpath = 'train/'+dftrain.loc[i]['img_file']
        bb = [dftrain.loc[i]['bbox_x1'], dftrain.loc[i]['bbox_y1'], dftrain.loc[i]['bbox_x2'], dftrain.loc[i]['bbox_y2']]
        cl = dftrain.loc[i]['class']
        y_train.append(cl)
        test1 = prepro_img(imgpath, bb, margin=20, size=(299,299))
        x_train.append(test1)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    
    ### shuffle
    idx = np.arange(len(x_train))
    np.random.shuffle(idx)
    x_train = x_train[idx]
    y_train = y_train[idx]    
    
    print(x_train.shape, y_train.shape)
    np.save('x_train3.npy', x_train)
    np.save('y_train3.npy', y_train)
else:
    print('x_train3.npy already exsits!')


# In[37]:


# 테스트 데이터도 전처리하여 저장한다.
# make test data numpy 
cnt_test = dftest.shape[0]
x_test=[]
if not os.path.exists('x_test3.npy') or force:
    for i in range(cnt_test):
        if i%1000==0 :
            print(i, '/', cnt_test)
        imgpath = 'test/'+dftest.loc[i]['img_file']
        bb = [dftest.loc[i]['bbox_x1'], dftest.loc[i]['bbox_y1'], dftest.loc[i]['bbox_x2'], dftest.loc[i]['bbox_y2']]
        test1 = prepro_img(imgpath, bb, margin=20, size=(299,299))
        x_test.append(test1)
    x_test = np.asarray(x_test)
    print(x_test.shape)
    np.save('x_test3.npy', x_test)
else:
    print('x_test3.npy already exsits!')


# In[41]:


# verify save file
ri = np.random.randint(0, 6000, 5)
print(ri)
plt.figure(figsize=(len(ri)*8, 8))
for a,i in enumerate(ri):
    plt.subplot(1,len(ri), a+1)
    plt.imshow(x_test[i])
plt.show()


# In[42]:


# release memory 
del dftrain
del dftest
del dfclass
del x_train
del y_train
del x_test


# In[ ]:





# In[ ]:




