from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(
    rescale=1./255
)

xy_train = train_datagen.flow_from_directory(
    'D:/_data/image/brain/train/', 
    target_size=(150,150),
    batch_size=1,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
)
# Found 160 images belonging to 2 classes.
xy_test = test_datagen.flow_from_directory(
    'D:/_data/image/brain/test/', 
    target_size=(150,150),
    batch_size=7,
    class_mode='binary',
    shuffle=True,
)
# Found 120 images belonging to 2 classes.
print(xy_train)
# <keras.preprocessing.image.DirectoryIterator object at 0x00000145D169A040>
# print(xy_train[33])
# ValueError: Asked to retrieve element 33, but the Sequence has length 32
# (5, 150, 150, 3)
# (5, 150, 150, 1)
print(xy_train[1][0].shape)
print(type(xy_train))
print(type(xy_train[0][1]))



