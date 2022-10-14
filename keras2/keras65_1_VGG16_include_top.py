import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.applications import VGG16

# model.trainable = False
model = VGG16(weights='imagenet',include_top=True,input_shape=(32,32,3))
model.summary()
print(len(model.weights))
print(len(model.trainable_weights))