# cnn 구성
# upsampling 추가 - 리얼리스트, 쌍샘플링 2가지

import numpy as np
from keras.datasets import mnist
(x_train, _), (x_test, _) = mnist.load_data()
print(x_train.shape)
x_train = x_train.reshape(60000, 28,28,1)/255
x_test = x_test.reshape(10000, 28,28,1)/255
from keras.models import Sequential, Model
from keras.layers import Dense,Input,Conv2D,UpSampling2D

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(hidden_layer_size, kernel_size=(1,1), input_shape=(28,28,1),activation='relu',padding='same'))
    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(hidden_layer_size, kernel_size=(16,16),activation='relu',padding='valid'))
    model.add(Conv2D(hidden_layer_size, kernel_size=(14,14),activation='relu',padding='valid'))
    model.add(Conv2D(1, kernel_size=(1,1),activation='sigmoid',padding='valid'))
    #아웃풋 노드를 히든레이어 사이즈만큼 넣겠다
    model.summary()
    return model

model = autoencoder(hidden_layer_size=64) #같은 차원축소 기능인 pca로 피처갯수조정

model.compile(optimizer='adam',loss='binary_crossentropy')

model.fit(x_train,x_train,epochs=10)

output = model.predict(x_test)

from matplotlib import pyplot as plt
import random
fig, ((ax1,ax2,ax3,ax4,ax5),(ax6,ax7,ax8,ax9,ax10)) = plt.subplots(2,5,figsize=(20,7))

#이미지 다섯개 무작위 고르기
random_images = random.sample(range(output.shape[0]),5)

#원본 이미지를 맨위에
for i, ax in enumerate([ax1,ax2,ax3,ax4,ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28,28),cmap='gray')
    if i ==0:
        ax.set_ylabel('input',size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 출력이미지
for i, ax in enumerate([ax6,ax7,ax8,ax9,ax10]):
    ax.imshow(output[random_images[i]].reshape(28,28),cmap='gray')
    if i ==0:
        ax.set_ylabel('output',size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()