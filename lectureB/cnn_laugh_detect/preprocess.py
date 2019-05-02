'''
preprocess dataset
'''
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2


base_path = 'dataset/'

train_data = h5py.File( os.path.join(base_path, 'train_happy.h5'), 'r')
x_train = np.array( train_data['train_set_x'][:])
y_train = np.array( train_data['train_set_y'][:])

test_data = h5py.File(os.path.join(base_path, 'test_happy.h5'), 'r')
x_test = np.array( test_data['test_set_x'][:])
y_test = np.array( test_data['test_set_y'][:])

# y 형식이 1차원 array임.

# 이것을 2차원 array의 컬럼 벡터 형태로 변경.
y_train = y_train.reshape((y_train.shape[0], 1))
y_test = y_test.reshape((y_test.shape[0], 1))
print(y_train)


plt.subplot(2,2,1)
plt.title(y_train[0])
plt.imshow(x_train[0])
plt.subplot(2,2,2)
plt.title(y_train[1])
plt.imshow(x_train[1])
plt.show()

# converting gray color

x_result=[]
for x in x_train:
    img = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    x_result.append(img)
x_result = np.array(x_result)
np.save('dataset/x_train.npy', x_result)
print('save : x_train ', x_result.shape)

x_result=[]
for x in x_test:
    img = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    x_result.append(img)
x_result = np.array(x_result)
np.save('dataset/x_test.npy', x_result)
print('save : x_test ', x_result.shape)

np.save('dataset/y_train.npy', y_train)
print('save : y_train ', y_train.shape)

np.save('dataset/y_test.npy', y_test)
print('save : y_test ', y_test.shape)


plt.subplot(2,2,1)
plt.title(y_test[0])
plt.imshow(x_result[0], cmap='gray')
plt.subplot(2,2,2)
plt.title(y_test[1])
plt.imshow(x_result[1], cmap='gray')
plt.show()
