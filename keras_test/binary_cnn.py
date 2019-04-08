
'''
이미지를 이진 분류

'''

# 0. 사용할 패키지 불러오기
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Dropout
import os


width = 28
height = 28

# 1. 데이터셋 생성하기

# 훈련셋과 시험셋 불러오기
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, width, height, 1).astype('float32') / 255.0
x_test = x_test.reshape(10000, width, height, 1).astype('float32') / 255.0

# 훈련셋과 검증셋 분리
x_val = x_train[50000:]
y_val = y_train[50000:]
x_train = x_train[:50000]
y_train = y_train[:50000]

# 데이터셋 전처리 : 홀수는 1, 짝수는 0으로 변환
y_train = y_train % 2
y_val = y_val % 2
y_test = y_test % 2



# 2. 모델 구성하기
if os.path.exists('binary_cnn.h5'):

    # model = load_model('binary_cnn.h5')

    # split loading test
    with open('binary_cnn_model.json', 'r') as f:
        jsonstr = f.read()
        model = model_from_json(jsonstr)
        model.load_weights('binary_cnn_weight.h5')
        model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

        W1 = model.layers[0].get_weights()[0]    # Conv2D(32,(3,3), relu, in=(w,h,1))
        b1 = model.layers[0].get_weights()[1]
        print("W1 shape0=", W1.shape, ' shape1=', b1.shape)
        # (3,2,1,32)  , bias=32
        # row,col, channel, kernel

        W2 = model.layers[1].get_weights()[0]
        b2 = model.layers[1].get_weights()[1]
        print("W2 shape0=", W2.shape, ' shape1=', b2.shape)
        # (3,2,32,32)  , bias=32


        W9 = model.layers[9].get_weights()[0]
        b9 = model.layers[9].get_weights()[1]
        print("W9 shape0=", W9.shape, ' shape1=', b9.shape)
        # (1024,256)  , bias=32

else:
    model = Sequential()
    model.add(Conv2D(32, (3, 2), activation='relu', input_shape=(width, height, 1)))
    model.add(Conv2D(32, (3, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # 3. 모델 학습과정 설정하기
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

    # 4. 모델 학습시키기
    hist = model.fit(x_train, y_train, epochs=1, batch_size=32, validation_data=(x_val, y_val))

    # 5. 학습과정 살펴보기
    # % matplotlib
    # inline
    import matplotlib.pyplot as plt

    fig, loss_ax = plt.subplots()

    acc_ax = loss_ax.twinx()

    loss_ax.plot(hist.history['loss'], 'y', label='train loss')
    loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
    loss_ax.set_ylim([0.0, 0.5])

    acc_ax.plot(hist.history['acc'], 'b', label='train acc')
    acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')
    acc_ax.set_ylim([0.8, 1.0])

    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    acc_ax.set_ylabel('accuray')

    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')

    plt.show()

    if not os.path.exists('binary_cnn.h5'):
        model.save('binary_cnn.h5')  # save all. (network model, weighs)

    # save split
    with open('binary_cnn_model.json', 'w') as f:
        f.write(model.to_json())

    model.save_weights('binary_cnn_weight.h5')


# 6. 모델 평가하기
if False:
    print('## evaluation loss and_metrics ##')
    loss_and_metrics = model.evaluate(x_test, y_test, batch_size=32)
    print(loss_and_metrics)

    # 7. 모델 사용하기
    yhat_test = model.predict(x_test, batch_size=32)

    # % matplotlib
    # inline
    import matplotlib.pyplot as plt

    plt_row = 5
    plt_col = 5

    plt.rcParams["figure.figsize"] = (10, 10)

    f, axarr = plt.subplots(plt_row, plt_col)
    print('## predict test ##')

    for i in range(plt_row * plt_col):
        sub_plt = axarr[int(i / plt_row), i % plt_col]
        sub_plt.axis('off')
        sub_plt.imshow(x_test[i].reshape(width, height))

        sub_plt_title = 'R: '

        if y_test[i]:
            sub_plt_title += 'odd '
        else:
            sub_plt_title += 'even '

        sub_plt_title += 'P: '

        if yhat_test[i] >= 0.5:
            sub_plt_title += 'odd '
        else:
            sub_plt_title += 'even '

        sub_plt.set_title(sub_plt_title)

    plt.show()
