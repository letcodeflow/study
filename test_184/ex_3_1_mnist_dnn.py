#파라미터 
#모델구현
#데이터
#학습 평가

from keras import layers, models

class DNN(models.Sequential):
    def __init__(self, Nin, Nh_l,Nout):
        super().__init__()
        self.add(layers.Dense(Nh_l[0], activation='relu', input_shape=(Nin,)))
        for i in range(10):
            self.add(layers.Dense(Nh_l[1], activation='relu'))
        self.add(layers.Dense(Nout, activation='softmax'))
        self.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

import numpy as np
from keras.datasets import fashion_mnist
from keras.utils import np_utils

def Data_func():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    Y_train = np_utils.to_categorical(y_train)
    Y_test = np_utils.to_categorical(y_test)

    L, W, H = X_train.shape
    X_train = X_train.reshape(-1, W*H)
    X_test = X_test.reshape(-1, W*H)

    X_train = X_train/255.
    X_test = X_test/255.

    return (X_train, Y_train), (X_test, Y_test)

from keras