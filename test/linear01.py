
import numpy as np
import matplotlib.pyplot as plt

#doesnt work
x_train = np.array([50,100,150,200])
y_train= np.array([150,100,50,0])

#work
# x_train = np.array([30,60,70,100])
# y_train= np.array([60,120,145,195])

def model(x,w,b):
    return x*w+b;

def cost(y,y_hat):
    return np.sum((y-y_hat)**2)/y.size

learning_rate=0.0001
def trainning_round(x_train,y_train,w,b,learning_rate):

    y_hat=model(x_train,w,b)
    j = cost(y_train,y_hat)

    # w_gradient=-2*x_train.dot(y_train-y_hat)
    # b_gradient=-2*np.sum(y_train-y_hat)
    w_gradient=x_train.dot(y_hat-y_train) / y_train.size
    b_gradient=np.sum(y_hat-y_train) / y_train.size

    print(w_gradient, b_gradient)

    w=w-learning_rate*w_gradient
    b=b-learning_rate*b_gradient
    print(j, w,b)
    return w,b

num_epoch=200000
def train(X,Y):

    w=2.1
    b=1.5

    #for plt
    ar = np.arange(0, 200, 0.5)

    for i in range(num_epoch):
        w,b=trainning_round(X,Y,w,b,learning_rate)

    plt.plot(ar,model(ar, w, b))
    plt.axis([0, 300, 0, 200])

    plt.plot(X, Y, 'ro')

train(x_train,y_train)
plt.show()


