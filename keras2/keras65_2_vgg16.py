import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.applications.vgg16 import VGG16

# model = VGG16() #include_top = True,input_shape=224,224,3
vgg16 = VGG16(weights='imagenet',include_top=False,input_shape=(32,32,3))

# vgg16.trainable=False 
# Total params: 14,766,998
# Trainable params: 52,310
# Non-trainable params: 14,714,688

# vgg16.summary()
model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10,activation='softmax'))
# model.trainable = False
model.summary()

print(len(model.weights))
print(len(model.trainable_weights))

# vgg false
# 30
# 4

# model false
# 30
# 0

# true true
# 30
# 30