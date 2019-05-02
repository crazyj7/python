
# denosing with autoencoder. MNIST

import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential, load_model
import pickle
import os

savefile='mnist.h5'
savemodel='dautoencoder.mod'

if os.path.exists(savefile) :
    print('loading from cache...')
    f = open(savefile, 'rb')
    x_train = pickle.load(f)
    x_test =  pickle.load(f)
else:
    print('downloading.. MNIST..')
    (x_train,_), (x_test, _) = mnist.load_data()
    f = open(savefile, 'wb')
    pickle.dump(x_train, f)
    pickle.dump(x_test, f)
    f.close()

print('x_train shape=', x_train.shape)  # 60000*28*28
print('x_test shape=', x_test.shape)    # 10000*28*28

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = np.reshape(x_train, (len(x_train), 784))
x_test = np.reshape(x_test, (len(x_test), 784))

# 훈련셋과 검증셋 분리
x_val = x_train[50000:]
x_train = x_train[:50000]


# add noise

noise_factor=0.5
x_train_noisy = x_train + noise_factor*np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_val_noisy = x_val + noise_factor*np.random.normal(loc=0.0, scale=1.0, size=x_val.shape)
x_test_noisy = x_test + noise_factor*np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0, 1)
x_val_noisy = np.clip(x_val_noisy, 0, 1)
x_test_noisy = np.clip(x_test_noisy, 0, 1)

print (x_train_noisy)

# draw image
if False:
    n=10
    plt.figure(figsize=(20,2))
    for i in range(n):
        ax = plt.subplot(1, n, (i+1))
        plt.imshow(x_test_noisy[i].reshape(28,28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


# make models
# if not os.path.exists(savemodel):
if True:
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=784))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(784, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy')

    model.fit(x_train_noisy, x_train, batch_size=256,
              nb_epoch=100, shuffle=True,
              validation_data=(x_val_noisy, x_val))
              # validation_data=(x_test_noisy, x_test))

    model.save(savemodel)
    print('save models...')
else :
    print('load models...')
    model = load_model(savemodel)


# evaluation

decoded_imgs = model.predict(x_test_noisy)
n=10
plt.figure(figsize=(20,6))
for i in range(n):
    # display original
    ax = plt.subplot(3, n, i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display noisy
    ax = plt.subplot(3, n, i+n+1)
    plt.imshow(x_test_noisy[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruct
    ax = plt.subplot(3, n, i+2*n+1)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()




