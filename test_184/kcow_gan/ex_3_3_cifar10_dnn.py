#data
#modeling

from keras import layers, models

class DNN(models.Sequential):
    def __init__(self, Nin, Nh_l,pd_l, Nout):
        super().__init__()
        self.add(layers.Dense(Nh_l[0], activation='relu', input_shape=(Nin,)))
        self.add(layers.Dropout(pd_l[0]))
        self.add(layers.Dense(Nh_l[1], activation='relu'))
        self.add(layers.Dropout(pd_l[1]))
        self.add(layers.Dense(Nout, activation='softmax'))
        self.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


import numpy as np
from keras.datasets import cifar100
from keras.utils import np_utils

def Data_func():
    (X_train, y_train), (X_test, y_test) = cifar100.load_data()

    Y_train = np_utils.to_categorical(y_train)
    Y_test = np_utils.to_categorical(y_test)

    L, W, H, C = X_train.shape
    print(X_train.shape)
    X_train = X_train.reshape(-1, W*H*C)
    X_test = X_test.reshape(-1, W*H*C)

    X_train = X_train/255.
    X_test = X_test/255.

    return (X_train, Y_train), (X_test, Y_test)
Data_func()
def main(Pd_l, Nh_l):
    Pd_l = [0.5,0.5]
    Nh_l = [500,200]
    number_of_class = 100
    Nout = number_of_class
    
    (X_train, Y_train),(X_test, Y_test) = Data_func()
    model = DNN(X_train.shape[1], Nh_l, Pd_l, Nout)
    model.fit(X_train, Y_train, epochs=100, batch_size=100, validation_split=0.2)

    print(model.evaluate(X_test, Y_test, batch_size=100))

if __name__=='__main__':
    for i in range(100):
        main(i for i in arange(10), z for z in arange(10))