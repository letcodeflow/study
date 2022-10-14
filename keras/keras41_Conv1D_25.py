from keras.datasets import cifar100
from matplotlib.cbook import flatten
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout, Conv2D, LSTM,Conv1D,MaxPool1D,Flatten
import pandas as pd
import numpy as np
#1.데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
print(y_train[:5])
print(np.unique(y_train,return_counts=True))
x_train = x_train.reshape(50000,32*32*3)
x_test = x_test.reshape(10000,32*32*3)
print(x_train.shape)
print(x_test.shape)
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
x_train = x_train.reshape(50000,32,96)
x_test = x_test.reshape(10000,32,96)
#2.모델
# model = Sequential()
# model.add(Dense(64, input_shape=(32*32*3,)))
# model.add(Dense(64, input_shape=(784,)))
# #x 쉐잎이 784 혹은 28*28이 되게한다
# model.add(Dense(100,activation='softmax'))

input1 = Input(shape=(32,96))
dense1 = Conv1D(100,12)(input1)
dense2 = MaxPool1D()(dense1)
dense3 = Dense(100)(dense2)
dense4 = Dense(100)(dense3)
# flat = Flatten()
output1 = Dense(100)(dense4)
flat = Flatten()(output1)
dense5 = Dense(100)(flat)
model = Model(inputs=input1, outputs=dense5)
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

# lstm
# 0.0116
