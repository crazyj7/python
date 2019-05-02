import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPool2D
from keras.layers import Activation, Dropout, Flatten, Dense
import os


dirname = os.path.dirname(__file__)
os.chdir(dirname)


# 분류 대상
category = ["chair", "camera", "butterfly"]
nb_classes = len(category)

image_w = 64
image_h = 64

x_train, x_test, y_train, y_test = np.load('./images/101.npy', allow_pickle=True)

# normalize
x_train = x_train.astype("float") / 256
x_test = x_test.astype("float") / 256
print('x shape', x_train.shape)

# make models
model = Sequential()
model.add(Convolution2D(32, kernel_size=3, strides=3, padding='same',
                        input_shape=(image_h, image_w, 3)))
model.add(Activation('relu'))
model.add(MaxPool2D((2,2)))
model.add(Dropout(rate=0.25))

model.add(Convolution2D(64, kernel_size=3, strides=3))
model.add(Activation('relu'))
model.add(MaxPool2D((2,2)))
model.add(Dropout(rate=0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(rate=0.5))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop',
              metrics=['accuracy']
              )
model.fit(x_train, y_train, batch_size=10, epochs=200)

# eval
score = model.evaluate(x_test, y_test)
print('loss=', score[0])
print('acc=', score[1])
model.save('img2vec.models')
print('ok')


