from keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    # shear_range=0.7,
    fill_mode='nearest'
)

datagen2 = ImageDataGenerator(rescale = 1./255)



argument_size = 40000
print(x_train.shape[0])

randidx = np.random.randint(x_train.shape[0], size = argument_size)

print(np.min(randidx), np.max(randidx))
print(type(randidx))
x_argumented = x_train[randidx].copy()
y_argumented = y_train[randidx].copy()

print(x_argumented.shape)
print(y_argumented.shape)
x_train = x_train.reshape(60000, 28,28,1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],1)

x_argumented = x_argumented.reshape(x_argumented.shape[0], x_argumented.shape[1], x_argumented.shape[2],1)

x_argumented = train_datagen.flow(x_argumented, y_argumented,
                                  batch_size=argument_size,
                                  shuffle=False,).next()[0]
print(x_argumented)
print(x_argumented.shape)


x_train = np.concatenate((x_train, x_argumented))
y_train = np.concatenate((y_train, y_argumented))

# x_argumented = datagen2.flow(x_train,y_train, batch_size=64, shuffle=False) #fit_generator를 쓰기 위함 이었지만 버전문제, fit에도 적용
# sparse_categorical_crossentropy 원핫 하지 않고 y는 그대로
# t= x_train[randidx+60000]
print(x_argumented.shape)
print(y_train.shape)

import matplotlib.pyplot as plt
plt.figure(figsize=(2,10))

# plt.imshow(x_argumented[3])
print(randidx[0])
for i in range(20):
    if i <= 9:
        plt.subplot(2, 10, i+1)
        plt.axis('off')
        plt.imshow(x_argumented[i], cmap='gray')
    else:
        plt.subplot(2, 10, i+1)
        plt.axis('off')
        plt.imshow(x_train[randidx[i-10]], cmap='gray')
    
plt.show() 
# plt.imshow(x_train[p+60000])
# plt.show() 

print(x_train[0].shape) #28,28
print(x_train[0].reshape(28*28).shape) #784,
print(np.tile(x_train[0].reshape(28*28), argument_size).reshape(-1,28,28,1).shape) #100,28,28,1
print(np.zeros(argument_size))
# print(np.zeros(argument_size).shape)
x_data = train_datagen.flow(np.tile(x_train[0].reshape(28*28), argument_size).reshape(-1,28,28,1), np.zeros(argument_size),
                            batch_size=argument_size,
                            shuffle=True,)
# next()사용
# print(x_data)
# print(x_data[0])
# print(x_data[0][0].shape) 
# print(x_data[0][1].shape)
#    [0.00000000e+00]
#    [0.00000000e+00]]]]
# (28, 28, 1)
# (28, 28, 1)

print(x_data)
print(x_data[0])
print(x_data[0][0].shape) 
print(x_data[0][1].shape)
#      [0.00000000e+00],
#          [0.00000000e+00]]]], dtype=float32), array([0., 0., 0., 0., 0., 0., 
# 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,   
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,   
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,   
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,   
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
# (100, 28, 28, 1)
# (100,)




import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7,7,i+1)
    plt.axis('off')
    # plt.imshow(x_data[0][i], cmap='gray')
    plt.imshow(x_data[0][0][i], cmap='gray') #next 사용
plt.show() 

from keras.models import Model, Input
from keras.layers import Dense, Conv2D, Flatten

in1     = Input(shape=(28,28,1))
conv1   = Conv2D(2,(14,14),activation='relu',name='donky')(in1)
flat1   = Flatten()(conv1)
dens1   = Dense(1)(flat1)
model   = Model(inputs=in1, outputs=dens1)
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(x_train,y_train, epochs=1, batch_size=10, validation_split=0.2)

loss = model.evaluate(x_test, y_test)
pred = model.predict(x_test)
print(np.unique(y_test))
print(pred)
print(y_test.shape, pred.shape)
 
# print(x_train[randidx])
# import matplotlib.pyplot as plt
# plt.figure(figsize=(2,10))
# for i in range(20):
#     if i <= 9:
#         plt.subplot(2, 10, i+1)
#         plt.axis('off')
#         plt.imshow(x_train[randidx][i], cmap='gray')
#     else:
#         plt.subplot(2, 10, i+1)
#         plt.axis('off')
#         plt.imshow(x_train[randidx][i+60000], cmap='gray')

# plt.show() 
# x_train[randidx]





# from sklearn.metrics import r2_score
# r2 = r2_score()