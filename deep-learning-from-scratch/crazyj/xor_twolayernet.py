# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 16:24:06 2017

@author: Junhee
"""

# coding: utf-8
import sys, os
sys.path.append(os.pardir)

import numpy as np
from two_layer_net import TwoLayerNet

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = \
    ( np.array([[0,0],  [0,1],  [1,0],  [1,1],
                [0.1, 0.1], [0.1, 1.1], [1.1, 0.1], [1.1, 1.1],
                [-0.1, -0.1], [-0.1, 0.9], [0.9, -0.1], [0.9, 0.9],
                [0.2, 0.2], [0.2, 1.2], [1.2, 0.2], [1.2, 1.2]]), \
            np.array([[1,0], [0,1], [0,1], [1,0], 
                      [1,0], [0,1], [0,1], [1,0],
                      [1,0], [0,1], [0,1], [1,0],
                      [1,0], [0,1], [0,1], [1,0]] ) 
            ),  \
    ( np.array([[0,0],[0,1],[1,0],[1,1]]), \
            np.array([[1,0], [0,1], [0,1], [1,0]] ) )
    
network = TwoLayerNet(input_size=2, hidden_size=2, output_size=2, weight_init_std=0.7)

iters_num = 1000
train_size = x_train.shape[0]
batch_size = 4
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    #batch_mask = np.random.choice(train_size, batch_size)
    batch_mask = range(0, 4)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 기울기 계산
    #grad = network.numerical_gradient(x_batch, t_batch) # 수치 미분 방식
    grad = network.gradient(x_batch, t_batch) # 오차역전파법 방식(훨씬 빠르다)
    
    # 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)

