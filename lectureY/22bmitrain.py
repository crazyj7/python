from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping
import pandas as pd, numpy as np

csv = pd.read_csv('bmi.csv')
maxweight = np.max( csv['weight'])
maxheight = np.max( csv['height'])
csv['weight'] /= maxweight
csv['height'] /= maxheight

bmi_class = {
    "thin":[1,0,0],
    "normal":[0,1,0],
    "fat":[0,0,1]
}
y = np.empty((20000,3))
for i,v in enumerate(csv['label']):
    y[i] = bmi_class[v]
print ( y[0:3] )

x=csv[["height", "weight"]].values
print(x[0:3])

x_train, y_train = x[:15000], y[:15000]
x_test, y_test = x[15000:], y[15000:]

# make model
model = Sequential()
model.add(Dense(512, input_shape=(2,)))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(3))
model.add(Activation('softmax'))

model.compile("rmsprop", "categorical_crossentropy", metrics=['accuracy'])

# train
# model.fit(x_train, y_train)
model.fit(x_train, y_train,
          batch_size=100,
          nb_epoch=20,
          validation_split=0.1,
          callbacks=[EarlyStopping(monitor='val_loss', patience=2)],
          verbose=1)

score = model.evaluate(x_test, y_test)
print()
print('loss=',score[0], 'acc=',score[1])


