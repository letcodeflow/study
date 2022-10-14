from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', input_shape=(28,28,1)))
model.add(MaxPooling2D())
model.add(Conv2D(7, (2,2), padding='valid', activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.summary()
# Layer (type)                 Output Shape              Param #   
# =================================================================
# conv2d (Conv2D)              (None, 28, 28, 64)        640       
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 14, 14, 64)        0
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 13, 13, 7)         1799
# _________________________________________________________________
# flatten (Flatten)            (None, 1183)              0
# _________________________________________________________________
# dense (Dense)                (None, 10)                11840
# =================================================================
# Total params: 14,279
# Trainable params: 14,279
# Non-trainable params: 0