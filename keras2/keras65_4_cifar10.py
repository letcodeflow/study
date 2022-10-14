
import numpy as np
import pandas as pd
from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input,Dropout,GlobalAveragePooling2D
from keras.applications.vgg16 import VGG16
#1. data
(x_train,y_train),(x_test,y_test) = cifar10.load_data()
# print(x_train.shape,x_test.shape) #(50000, 32, 32, 3) (10000, 32, 32, 3)
x_train = x_train.reshape(-1,32*32*3)
x_test = x_test.reshape(-1,32*32*3)

from sklearn.preprocessing import StandardScaler,MinMaxScaler
scaler = MinMaxScaler()
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train).reshape(-1,32,32,3)
x_test = scaler.transform(x_test).reshape(-1,32,32,3)

from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train).reshape(-1,10)
y_test = to_categorical(y_test).reshape(-1,10)
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.applications.vgg16 import VGG16

# model = VGG16() #include_top = True,input_shape=224,224,3
vgg16 = VGG16(weights='imagenet',include_top=False,input_shape=(32,32,3))
print(vgg16.trainable_weights) #(3, 3, 3, 64)
vgg16.trainable=False 
# Total params: 14,766,998
# Trainable params: 52,310
# Non-trainable params: 14,714,688

# vgg16.summary()
model = Sequential()
model.add(vgg16)
model.add(GlobalAveragePooling2D())
model.add(Dense(100))
model.add(Dense(10,activation='softmax'))
# model.trainable = False
# model.summary()

# print(len(model.weights))
# print(len(model.trainable_weights))

print(model.layers)
import pandas as pd
pd.set_option('max_colwidth',-1)
layers= [(layer,layer.name,layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers,columns=['layer Type','Layer Name','Layer Trainable'])
print(results)

#2.model
# def build_model(drop=0.5,optimizer='adam',activation='relu',node1=512):

# vgg16 = VGG16(include_top=False,weights='imagenet',input_shape=(32,32,3))



# activation='relu'
# drop = 0.3
# vgg16 = VGG16(include_top=False,weights='imagenet',input_shape=(32,32,3))
# vgg16.trainable=False


# x = GlobalAveragePooling2D()(vgg16)
# x = Dense(100, activation=activation,name='hidden3')(x)
# x = Dropout(drop)(x)
# outputs = Dense(100, activation='softmax',name='outputs')(x)

# import tensorflow as tf
from keras.optimizers import adam_v2
learning_rate = 0.01
optimizer = adam_v2.Adam(lr=learning_rate)
# model = Model(inputs=vgg16,outputs=outputs)
# model.trainable_weights=False
model.compile(optimizer=optimizer,metrics=['accuracy'], loss = 'categorical_crossentropy')
# model.summary()

print(model.weights)
print(model.trainable_weights)
    # return model
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
es = EarlyStopping(monitor='val_lostt',patience=20,mode='min',verbose=1,)
reduce_lr = ReduceLROnPlateau(monitor='val_lostt',patience=10,verbose=1,mode='min',min_lr=0,factor=0.0005)
import time
start = time.time()
model.fit(x_train,y_train,epochs=100,validation_split=0.3,callbacks=[es,reduce_lr])
end = time.time()
# loss, acc = model.evaluate(x_test,y_test)
print('time',end-start,'cho')
loss, acc = model.evaluate(x_test,y_test)
from sklearn.metrics import accuracy_score
pred = model.predict(x_test)
print(y_test.shape,pred.shape)
print('acc',accuracy_score(np.argmax(y_test,axis=1),np.argmax(pred,axis=1)))
