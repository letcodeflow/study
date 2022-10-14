
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd

train_datagen = ImageDataGenerator(
    rescale=1./255,
    # horizontal_flip=True,
    # vertical_flip=True,
    # width_shift_range=0.1,
    # height_shift_range=0.1,
    # rotation_range=5,
    # zoom_range=1.2,
    # shear_range=0.7,
    # fill_mode='nearest'
)
test_datagen = ImageDataGenerator(
    rescale=1./255
)

xy_train = train_datagen.flow_from_directory(
    'D:/study_data/_data/image/brain/train/', 
    target_size=(150,150),
    batch_size=500,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
)
# Found 160 images belonging to 2 classes.
xy_test = test_datagen.flow_from_directory(
    'D:/study_data/_data/image/brain/test/', 
    target_size=(150,150),
    batch_size=500,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
)
# Found 120 images belonging to 2 classes.
print(xy_train)
# <keras.preprocessing.image.DirectoryIterator object at 0x00000145D169A040>
# print(xy_train[33])
# ValueError: Asked to retrieve element 33, but the Sequence has length 32
# (5, 150, 150, 1)
print(xy_train[0][0].shape,xy_train[0][1].shape)
print(xy_test[0][0].shape,xy_test[0][1].shape)
# print(type(xy_train))
# print(type(xy_train[0][1]))
np.save('D:/study_data/_save/_npy/keras47_5_train_x.npy', arr=xy_train[0][0])
np.save('D:/study_data/_save/_npy/keras47_5_train_y.npy', arr=xy_train[0][1])
np.save('D:/study_data/_save/_npy/keras47_5_test_x.npy', arr=xy_test[0][0])
np.save('D:/study_data/_save/_npy/keras47_5_test_y.npy', arr=xy_test[0][1])
# 현재 5,200,200,1 데이터셋이 32개
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