# 0. 사용할 패키지 불러오기
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from keras.models import load_model

# 1. 데이터셋 생성하기
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255.0
x_test = x_test.reshape(10000, 784).astype('float32') / 255.0
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

xhat_idx = np.random.choice(x_test.shape[0], 5)
xhat = x_test[xhat_idx]

# 2. 모델 구성하기
model = load_model('mnist_mlp_model.h5')

yhat = model.predict_classes(xhat)

for i in range(5):
    print('True : ' + str(np.argmax(y_test[xhat_idx[i]]))+', Predict:'+str(yhat[i]))


# mode view
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

# %matplotlib inline

SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))


