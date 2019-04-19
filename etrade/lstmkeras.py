import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, Dense, Activation

import os


import datetime
import time


data = pd.read_csv('datasets/005930.KS.csv')
data.head() # 앞에거 5개 출력

high_prices = data['High'].values
low_prices = data['Low'].values
mid_prices = (high_prices+low_prices)/2


# create window
seq_len=50
sequence_length = seq_len+1

result=[]
for index in range(len(mid_prices)-sequence_length):
    result.append(mid_prices[index:index+sequence_length])

# normalize
normalized_data=[]
for window in result:
    normalized_window = [((float(p)/float(window[0]))-1) for p in window ]
    normalized_data.append(normalized_window)
result = np.array(normalized_data)

# split train
row = int(round(len(result))*0.9)
train = result[:row, :]
np.random.shuffle(train)

x_train=train[:,:-1]
x_train=np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
y_train = train[:,-1]


x_test= result[row:,:-1]
x_test=np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
y_test= result[row:,-1]

x_train.shape, y_train.shape
x_test.shape, y_test.shape

if not os.path.exists('lstmkeras.model') :
    # build model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(50,1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='rmsprop')

    model.summary()

    # train
    model.fit(x_train, y_train,
              validation_data=(x_test, y_test),
              batch_size=10,
              epochs=20)
    model.save('lstmkeras.model')
else:
    model = load_model('lstmkeras.model')

pred = model.predict(x_test)
print(y_test)
print(pred)

fig = plt.figure()
ax=fig.add_subplot(111)
ax.plot(y_test, label='True')
ax.plot(pred, label='Prediction')
ax.legend()
plt.show()



