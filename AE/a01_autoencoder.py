import numpy as np
from keras.datasets import mnist
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float32')/255.
x_test = x_test.reshape(10000, 784).astype('float32')/255.
#타입변환형 혹시나해서 추가하는 것이므로 어떻게 쓰든 상관은 없다 - 스케일링...

from keras.models import Sequential, Model
from keras.layers import Dense,Input

input_img = Input(shape=(784,))
encoded = Dense(64,activation='relu')(input_img)
#자잘한 특성제거, 활성화 함수는 아무거나 써도 된다.
#노드수나 층수는 결과값을 보면서 판단해야 한다 활성화함수도
#마지막 활성화 함수 바꿀때 로스함수도 고려한다
decoded = Dense(784, activation='sigmoid')(encoded)
#다시 원래크기로 불리기, .sigmoid 위에서 스케일링 했기때문에 같은 수치로 맞춰준 것
autoencoder = Model(input_img, decoded)
autoencoder.summary()

autoencoder.compile(optimizer='adam',loss='binary_crossentropy', metrics=['acc'])

autoencoder.fit(x_train, x_train,epochs=30, batch_size=128, validation_split=0.2 )
#준지도 학습

decoded_img = autoencoder.predict(x_test)

import matplotlib.pyplot as plt
n=10
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2,n,i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2,n,i+1+n)
    plt.imshow(decoded_img[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
