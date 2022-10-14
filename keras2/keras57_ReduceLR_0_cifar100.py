
import numpy as np
import pandas as pd
from keras.datasets import cifar100
from sympy import Max
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input,Dropout,GlobalAveragePooling2D
import keras
#1. data
(x_train,y_train),(x_test,y_test) = cifar100.load_data()
print(x_train.shape,x_test.shape) #(50000, 32, 32, 3) (10000, 32, 32, 3)
x_train = x_train.reshape(-1,32*32*3)
x_test = x_test.reshape(-1,32*32*3)

from sklearn.preprocessing import StandardScaler,MinMaxScaler
scaler = MinMaxScaler()
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train).reshape(-1,32,32,3)
x_test = scaler.transform(x_test).reshape(-1,32,32,3)

from keras.utils import to_categorical
y_train = to_categorical(y_train).reshape(-1,100)
y_test = to_categorical(y_test).reshape(-1,100)

#2.model
# def build_model(drop=0.5,optimizer='adam',activation='relu',node1=512):
activation='relu'
node1=128
drop = 0.3
optimizer='adam'
inputs = Input(shape=(32,32,3),name='input')
x = Conv2D(node1, kernel_size=(2,2),padding='valid',activation=activation,name='hidden1')(inputs)
x = Dropout(drop)(x)
x = Conv2D(64, kernel_size=(2,2),padding='same',activation=activation,name='hidden5')(x)
x = Dropout(drop)(x)
x = MaxPool2D()(x)
x = Conv2D(32, kernel_size=(3,3),padding='valid',activation=activation,name='hidden6')(x)
x = Dropout(drop)(x)
# x = Flatten()(x)
x = GlobalAveragePooling2D()(x)
# x = Dense(256, activation=activation,name='hidden2')(x)
# x = Dropout(drop)(x)
x = Dense(100, activation=activation,name='hidden3')(x)
x = Dropout(drop)(x)
outputs = Dense(100, activation='softmax',name='outputs')(x)

model = Model(inputs=inputs,outputs=outputs)
model.compile(optimizer=optimizer,metrics=['accuracy'], loss = 'categorical_crossentropy')
model.summary()
    # return model
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
es = EarlyStopping(monitor='val_lostt',patience=20,mode='min',verbose=1,)
reduce_lr = ReduceLROnPlateau(monitor='val_lostt',patience=10,verbose=1,mode='min',min_lr=0,factor=0.0005)
import time
start = time.time()
model.fit(x_train,y_train,epochs=1,validation_split=0.3,callbacks=[es,reduce_lr])
end = time.time()

# loss, acc = model.evaluate(x_test,y_test)
print('time',end-start,'cho')
loss, acc = model.evaluate(x_test,y_test)
from sklearn.metrics import accuracy_score
pred = model.predict(x_test)
print(y_test.shape,pred.shape)
# print(y_test)
print(pred)
print('acc',accuracy_score(np.argmax(y_test,axis=1),np.argmax(pred,axis=1)))
