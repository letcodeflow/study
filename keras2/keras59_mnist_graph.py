from keras.datasets import mnist
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout, Conv2D,LSTM,Conv1D,MaxPool1D,Flatten
import pandas as pd
import numpy as np
#1.데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train[:10])

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
print(y_train[:5])
print(np.unique(y_train))
x_train = x_train.reshape(60000,14,56)
x_test = x_test.reshape(10000,14,56)
print(x_train.shape)
print(x_test.shape)
from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
#2.모델
# model = Sequential()
# model.add(Dense(64, input_shape=(28*28,)))
# model.add(Dense(64, input_shape=(784,)))
# #x 쉐잎이 784 혹은 28*28이 되게한다a
# model.add(Dense(10,activation='softmax'))


input1 = Input(shape=(14,56))
lstm1 = Conv1D(100,4)(input1)
dense2 = MaxPool1D()(lstm1)
dense3 = Dense(100)(dense2)
dense4 = Dense(100)(dense3)
flat = Flatten()(dense4)
output1 = Dense(10)(flat)
model = Model(inputs=input1, outputs=output1)
model.summary()


#3.컴파일 훈련
from keras.callbacks import EarlyStopping,ReduceLROnPlateau

es = EarlyStopping(monitor='val_loss',mode='min',patience=20,verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',mode='auto',patience=10,factor=0.00005,verbose=1)
from tensorflow.python.keras.optimizer_v2.adam import Adam
# from tensorflow.python.keras.optimizers import adam_v2
learning_rate = 0.01
optimizer = Adam(learning_rate=learning_rate)
import tensorflow as tf

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),optimizer=optimizer,metrics=['acc'])
import time
start = time.time()
hist = model.fit(x_train,y_train,epochs=50,batch_size=32,callbacks=[es,reduce_lr],validation_split=0.2,verbose=1)
end = time.time()-start

loss, acc = model.evaluate(x_test,y_test)
print('lr{},loss{:.4f}acc{:.4f}time{:.4f}'.format(learning_rate,loss,acc,end))

import matplotlib.pyplot as plt
plt.figure(figsize=(10,8))

plt.subplot(2,1,1)
plt.plot(hist.history['loss'],marker='.',label='loss')
plt.plot(hist.history['val_loss'],marker='.',label='val_loss')
plt.grid()
plt.title('loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='upper right')

plt.subplot(2,1,2)
plt.plot(hist.history['acc'],marker='.',label='loss')
plt.plot(hist.history['val_acc'],marker='.',label='val_loss')
plt.grid()
plt.title('acc')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.legend(['acc','val_acc'])

plt.show()