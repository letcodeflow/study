
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
# np.save('D:/study_data/_save/_npy/keras47_5_train_x.npy', arr=xy_train[0][0])
# np.save('D:/study_data/_save/_npy/keras47_5_train_y.npy', arr=xy_train[0][1])
# np.save('D:/study_data/_save/_npy/keras47_5_test_x.npy', arr=xy_test[0][0])
# np.save('D:/study_data/_save/_npy/keras47_5_test_y.npy', arr=xy_test[0][1])
# 현재 5,200,200,1 데이터셋이 32개
x_train = np.load('D:/study_data/_save/_npy/keras47_5_train_x.npy')
y_train = np.load('D:/study_data/_save/_npy/keras47_5_train_y.npy')
x_test = np.load('D:/study_data/_save/_npy/keras47_5_test_x.npy')
y_test = np.load('D:/study_data/_save/_npy/keras47_5_test_y.npy')
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
"""
#2.모델구성
from keras.models import Model, Input
from keras.layers import Dense, Conv2D,Flatten

in1 = Input(shape=(100,100,1))
conv2d1 = Conv2D(32, (2,2), activation='relu')(in1)
conv2d2 = Conv2D(64, (2,3), activation='relu')(conv2d1)
flat1   = Flatten()(conv2d2)
dens1   = Dense(16, activation='relu')(flat1)
dens2   = Dense(1, activation='sigmoid')(dens1)
model = Model(inputs = in1, outputs = dens2)
model.summary()
model.compile(loss = 'binary_crossentropy', optimizer= 'adam', metrics=['accuracy'])
# model.fit(xy_train[0][0], xy_train[0][1]) 배치 최대일때 가능
hist = model.fit_generator(xy_train, epochs = 10, 
                           validation_data=xy_test, validation_steps=4,
                           steps_per_epoch=3,)

acc = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print(acc)
print(val_accuracy[-1])
print(loss[-1])
print(val_loss[-1])

import matplotlib.pyplot as plt
plt.figure(figsize=(9,6))
plt.plot(hist.history['val_accuracy'])
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.xlabel('wj')
plt.ylabel('af')
plt.legend(loc ='lower center')
plt.title('dkjw')
plt.grid()
plt.gray()
plt.show() """