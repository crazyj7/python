import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, Dense, Input, MaxPooling2D, Activation, LeakyReLU, Flatten
from keras.models import Model, Sequential, load_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns


plt.style.use('dark_background')
basepath = 'dataset'

x_train = np.load(os.path.join(basepath, 'x_train.npy'))
y_train = np.load(os.path.join(basepath, 'y_train.npy'))
x_test = np.load(os.path.join(basepath, 'x_test.npy'))
y_test = np.load(os.path.join(basepath, 'y_test.npy'))

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# make image matrix to 3D
x_train = x_train.reshape((-1,64,64,1))
x_test = x_test.reshape((-1,64,64,1))

x_test_input = x_test.copy().astype(np.float64)
x_test_input -= np.mean(x_test, keepdims = True)
x_test_input /= (np.std(x_test, keepdims=True)+1e-6)

model = load_model('models/happy.h5')

y_pred = model.predict(x_test_input)
y_pred_log = (y_pred>0.5).astype(np.int)

print('test acc:%s'%accuracy_score(y_test, y_pred_log))

cm = confusion_matrix(y_test, y_pred_log)
sns.heatmap(cm, annot=True)
plt.show()

for i in range(9):
    plt.subplot(3,3,(i+1))
    idx = np.random.randint(0, len(x_test))
    plt.title(str(y_test[idx])+' P'+str(y_pred_log[idx]))
    plt.imshow(x_test[idx].reshape((64,64)), cmap='gray')

plt.show()

