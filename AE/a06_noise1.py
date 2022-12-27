import numpy as np
from keras.datasets import mnist
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float32')/255.
x_test = x_test.reshape(10000, 784).astype('float32')/255.

x_train_noised = x_train + np.random.normal(0,0.1,size=x_train.shape)
x_test_noised = x_test + np.random.normal(0,0.1,size=x_test.shape)
# 그림 진하게 만들기 
#결과값 0~1 -> 0~1.1
#1.1 컷..
x_train_noised = np.clip(x_train_noised,a_min=0,a_max=1)
x_test_noised = np.clip(x_test_noised,a_min=0,a_max=1)






from keras.models import Sequential, Model
from keras.layers import Dense,Input

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784,),activation='relu'))
    #아웃풋 노드를 히든레이어 사이즈만큼 넣겠다
    model.add(Dense(units=784, activation='sigmoid'))
    return model

model = autoencoder(hidden_layer_size=154) #같은 차원축소 기능인 pca로 피처갯수조정

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])

model.fit(x_train_noised,x_train,epochs=10)
#노이즈 있는 걸로 없는 걸 생성 예측하는 게 목표
output = model.predict(x_test_noised)

from matplotlib import pyplot as plt
import random
fig, ((ax1,ax2,ax3,ax4,ax5),(ax6,ax7,ax8,ax9,ax10),(ax11,ax12,ax13,ax14,ax15)) = plt.subplots(3,5,figsize=(20,7))

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
    ax.imshow(x_test_noised[random_images[i]].reshape(28,28),cmap='gray')
    if i ==0:
        ax.set_ylabel('noise',size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i, ax in enumerate([ax11,ax12,ax13,ax14,ax15]):
    ax.imshow(output[random_images[i]].reshape(28,28),cmap='gray')
    if i ==0:
        ax.set_ylabel('output',size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()