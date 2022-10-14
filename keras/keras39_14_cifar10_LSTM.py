from keras.datasets import cifar10
from matplotlib.cbook import flatten
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout, Conv2D, LSTM
import pandas as pd
import numpy as np
#1.데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
print(y_train[:5])
print(np.unique(y_train))
x_train = x_train.reshape(50000,32*32*3)
x_test = x_test.reshape(10000,32*32*3)
print(x_train.shape)
print(x_test.shape)
from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
x_train = x_train.reshape(50000,32,96)
x_test = x_test.reshape(10000,32,96)


#2.모델
# model = Sequential()
# model.add(Dense(64, input_shape=(32*32*3,)))
# model.add(Dense(64, input_shape=(784,)))
# #x 쉐잎이 784 혹은 28*28이 되게한다
# model.add(Dense(10,activation='softmax'))


input1 = Input(shape=(32,96))
lstm1 = LSTM(100)(input1)
dense2 = Dense(100)(lstm1)
dense3 = Dense(100)(dense2)
dense4 = Dense(100)(dense3)
output1 = Dense(10, activation='softmax')(dense4)
model = Model(inputs=input1, outputs=output1)
model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=20000)

#평가예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print(y_test)
print(y_predict)

y_predict = np.argmax(y_predict, axis= 1)
y_test = np.argmax(y_test, axis= 1)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predict)
print(acc)


# lstm1
# 0.2996