from enum import auto
import numpy as np
from keras.datasets import mnist
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float32')/255.
x_test = x_test.reshape(10000, 784).astype('float32')/255.
from keras.models import Sequential, Model
from keras.layers import Dense,Input

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784,),activation='relu'))
    #아웃풋 노드를 히든레이어 사이즈만큼 넣겠다
    model.add(Dense(units=784, activation='sigmoid'))
    return model

model_01 = autoencoder(hidden_layer_size=1)
model_04 = autoencoder(hidden_layer_size=4)
model_16 = autoencoder(hidden_layer_size=16)
model_32 = autoencoder(hidden_layer_size=32)
model_64 = autoencoder(hidden_layer_size=64)
model_128 = autoencoder(hidden_layer_size=128)

model_01.compile(optimizer='adam',loss ='binary_crossentropy')
model_04.compile(optimizer='adam',loss ='binary_crossentropy')
model_16.compile(optimizer='adam',loss ='binary_crossentropy')
model_32.compile(optimizer='adam',loss ='binary_crossentropy')
model_64.compile(optimizer='adam',loss ='binary_crossentropy')
model_128.compile(optimizer='adam',loss ='binary_crossentropy')


model_01.fit(x_train, x_train, epochs=10)
model_04.fit(x_train, x_train, epochs=10)
model_16.fit(x_train, x_train, epochs=10)
model_32.fit(x_train, x_train, epochs=10)
model_64.fit(x_train, x_train, epochs=10)
model_128.fit(x_train, x_train, epochs=10)

output_01 = model_01.predict(x_test)
output_04 = model_04.predict(x_test)
output_16 = model_16.predict(x_test)
output_32 = model_32.predict(x_test)
output_64 = model_64.predict(x_test)
output_128 = model_128.predict(x_test)

#레이어 갯수조절에 따른 결과값 차이

import matplotlib.pyplot as plt
import random

fig, axes = plt.subplots(7,5,figsize=(15,15))

random_imgs = random.sample(range(output_01.shape[0]),5)
outputs = [x_test,output_01,
output_04,
output_16,
output_32,
output_64,
output_128]

for row_num, row in enumerate(axes):
    for col_num, ax in enumerate(row):
        ax.imshow(outputs[row_num][random_imgs[col_num]].reshape(28,28),cmap='gray')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

plt.show()
