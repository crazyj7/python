import keras, datetime
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.applications import mobilenetv2
import numpy as np

import os
from keras.models import load_model

img_size = 224

modelsavefile = 'cat_hipster2.model'
output_size=18  #9points

data_00 = np.load('./cats/CAT_00_2.npy')
data_01 = np.load('./cats/CAT_01_2.npy')
data_02 = np.load('./cats/CAT_02_2.npy')
data_03 = np.load('./cats/CAT_03_2.npy')

data_06 = np.load('./cats/CAT_06_2.npy')

x_train = np.concatenate( (data_00.item().get('img2'),
                           data_01.item().get('img2'),
                           data_02.item().get('img2'),
                           data_03.item().get('img2')), axis=0)
y_train = np.concatenate( (data_00.item().get('lmk2'),
                           data_01.item().get('lmk2'),
                           data_02.item().get('lmk2'),
                           data_03.item().get('lmk2')), axis=0)

x_test = np.array( data_06.item().get('img2'))
y_test = np.array( data_06.item().get('lmk2'))

# normalize
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (-1, img_size, img_size, 3))
x_test = np.reshape(x_test, (-1, img_size, img_size, 3))

# y data
y_train = np.reshape(y_train, (-1, output_size))
y_test = np.reshape( y_test, (-1, output_size))

inputs = Input(shape=(img_size, img_size, 3))

if os.path.exists(modelsavefile):
    model = load_model(modelsavefile)
    print('continous train....')
else:
    mobilenetv2_model = mobilenetv2.MobileNetV2(input_shape=(img_size, img_size, 3),
                          alpha=1.0, include_top=False, weights='imagenet',
                        input_tensor=inputs, pooling='max')

    net = Dense(128, activation='relu')(mobilenetv2_model.layers[-1].output)
    net = Dense(64, activation='relu')(net)
    net = Dense(output_size, activation='linear')(net)

    model = Model(inputs=inputs, outputs=net)

    model.summary()

    # train
    model.compile(optimizer=keras.optimizers.Adam(), loss='mse')

# epoch 50
model.fit(x_train, y_train, epochs=5, batch_size=32, shuffle=True,
          validation_data=(x_test, y_test), verbose=1)

model.save(modelsavefile)




