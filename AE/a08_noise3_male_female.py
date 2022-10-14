#image 에 본인사진 넣고 predictfrom keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
# run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
datagen = ImageDataGenerator(
    rescale = 1./255
)
me = datagen.flow_from_directory('C:/Users/aiapalm\Downloads\me/',target_size=(100,100),
                                     batch_size=3000000000,
                                     class_mode='binary',
                                     shuffle=True,)
np.save('D:\study_data\_save\_npy\me.npy',arr = me[0][0])



# train = datagen.flow_from_directory('D:\study_data\_data\image/MENWOMEN/train/',target_size=(300,300),
#                                      batch_size=3000000000,
#                                      class_mode='binary',
#                                      shuffle=True,)
# test = datagen.flow_from_directory('D:\study_data\_data\image\MENWOMEN/test/',target_size=(300,300),
#                                      batch_size=300000000000,
#                                      class_mode='binary',
#                                      shuffle=True,)

# np.save('D:\study_data\_save\_npy\keras47_mwtrain_x', arr = train[0][0])
# np.save('D:\study_data\_save\_npy\keras47_mwtrain_y',arr = train[0][1])
# np.save('D:\study_data\_save\_npy\keras47_mwtest_x',arr = train[0][0])
# np.save('D:\study_data\_save\_npy\keras47_mwtest_y.npy',arr = train[0][1])

me = np.load('D:\study_data\_save\_npy\me.npy')
x_train = np.load('D:\study_data\_save\_npy\keras47_mwtrain_x.npy')
# y_train = np.load('D:\study_data\_save\_npy\keras47_mwtrain_y.npy')/255
x_test = np.load('D:\study_data\_save\_npy\keras47_mwtest_x.npy')
# y_test = np.load('D:\study_data\_save\_npy\keras47_mwtest_y.npy')/255
# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape) 
x_train_noised = x_train + np.random.normal(0,0.1,size=x_train.shape)
x_test_noised = x_test + np.random.normal(0,0.1,size=x_test.shape)
me_noised = me + np.random.normal(0,0.1,size=me.shape)
x_train_noised = np.clip(x_train_noised,a_min=0,a_max=1)
x_test_noised = np.clip(x_test_noised,a_min=0,a_max=1)
me_noised = np.clip(me_noised,a_min=0,a_max=1)
print(x_train.shape)
print(me.shape)
from keras.models import Model, Input
from keras.layers import Dense, Conv2D, Flatten, UpSampling2D
# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

in1     = Input(shape=(100,100,3))
Con2d1  = Conv2D(64,(2,2),padding='same', activation='relu',strides=2)(in1)
Con2d2 = UpSampling2D((2,2))(Con2d1)
# Con2d2  = Conv2D(2,(18,18),padding='same', activation='relu',strides=2)(Con2d1)
# Con2d2  = Conv2D(3,(15,15),padding='same', activation='relu')(Con2d2)
Con2d2  = Conv2D(3,(10,10),padding='same', activation='sigmoid')(Con2d2)
# flat1   = Flatten()(Con2d2)
# dens1   = Dense(1, activation='sigmoid')(flat1)

model   = Model(inputs = in1, outputs = Con2d2)
model.summary()

model.compile(loss='mse', optimizer='adam', metrics='accuracy')
model.fit(x_train_noised, x_train, epochs=10, batch_size=12, validation_split=0.2)

predme = model.predict(me)
output = model.predict(x_test_noised)

# print(pred)

from matplotlib import pyplot as plt
import random
fig, ((ax1,ax2,ax3,ax4,ax5),(ax6,ax7,ax8,ax9,ax10),(ax11,ax12,ax13,ax14,ax15)) = plt.subplots(3,5,figsize=(20,7))

#이미지 다섯개 무작위 고르기
# print(output.shape[0],x_test.shape,x_test_noised.shape)
random_images = random.sample(range(output.shape[0]),5)
# print(list(range(output.shape[0])))
# print(random_images[3])
#원본 이미지를 맨위에
for i, ax in enumerate([ax1,ax2,ax3,ax4,ax5]):
    ax.imshow(x_test[random_images[i]].reshape(100,100,3))
    if i ==0:
        ax.set_ylabel('input',size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
ax.imshow(me.reshape(100,100,3))

# 출력이미지
for i, ax in enumerate([ax6,ax7,ax8,ax9,ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(100,100,3))
    if i ==0:
        ax.set_ylabel('noise',size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
ax.imshow(me_noised.reshape(100,100,3))

for i, ax in enumerate([ax11,ax12,ax13,ax14,ax15]):
    ax.imshow(output[random_images[i]].reshape(100,100,3))
    if i ==0:
        ax.set_ylabel('output',size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
ax.imshow(predme.reshape(100,100,3))


# for i, ax in enumerate([ax5,ax10,ax15]):
#     ax.imshow(me[random_images[i]].reshape(50,50,3),cmap='gray')
#     if i ==0:
#         ax.set_ylabel('output',size=20)
#     ax.grid(False)
#     ax.set_xticks([])
#     ax.set_yticks([])
# for i, ax in enumerate([ax16,ax17,ax18,ax19,ax20]):
#     ax.imshow(me[random_images[i]].reshape(50,50,3))
#     if i ==0:
#         ax.set_ylabel('output',size=20)
#     ax.grid(False)
#     ax.set_xticks([])
#     ax.set_yticks([])
    
# for i, ax in enumerate([ax21,ax21,ax22,ax23,ax24]):
#     ax.imshow(predme[random_images[i]].reshape(50,50,3))
#     if i ==0:
#         ax.set_ylabel('output',size=20)
#     ax.grid(False)
#     ax.set_xticks([])
#     ax.set_yticks([])

plt.tight_layout()
plt.show()