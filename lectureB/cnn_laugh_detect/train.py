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

train_datagen = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    brightness_range=[0.5, 1.5],
    zoom_range=[0.8, 1.1],
    rotation_range=15.,
    channel_shift_range=25,
    horizontal_flip=True,
    data_format='channels_last'
)


test_datagen = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    data_format='channels_last'
)

train_batch_gen = train_datagen.flow(x_train, y_train, batch_size=16, shuffle=True)
test_batch_gen = test_datagen.flow(x_test, y_test, batch_size=16, shuffle=False)


inputs = Input(shape=(64,64,1))

net = Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(inputs)
net = LeakyReLU()(net)
net = Flatten()(net)

net = Dense(units=64)(net)
net = Activation('relu')(net)

net = Dense(units=1)(net)
outputs = Activation('sigmoid')(net)

model = Model(inputs = inputs, outputs=outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model.summary()

## train
# steps는 data 수 / 배치 사이즈.
model.fit_generator(
    train_batch_gen,
    steps_per_epoch=train_batch_gen.__len__(),  ## bug?
    epochs=20,
    validation_data=test_batch_gen,
    validation_steps=test_batch_gen.__len__(),  ## bug?
    callbacks=[
        ModelCheckpoint('models/happy.h5', monitor='val_acc', save_best_only=True, mode='max', verbose=1),
        ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, verbose=1, mode='auto', min_lr=1e-05)
    ]
)








