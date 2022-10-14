from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(3,3), padding='same', input_shape=(28,28,1))) 
# conv2d (Conv2D)              (None, 8, 8, 10)          100 유지
model.add(MaxPooling2D())
# 맥스풀링은 반으로줄이나?
model.add(Conv2D(7, (2,2), padding='same', activation='relu'))
# conv2d_1 (Conv2D)            (None, 7, 7, 7)           287
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax')) 
model.summary()
# Model: "sequential"
# _________________________________________________________________
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
# dense (Dense)                (None, 32)                37888
# _________________________________________________________________
# dense_1 (Dense)              (None, 32)                1056
# _________________________________________________________________
# dense_2 (Dense)              (None, 10)                330
# =================================================================
# Total params: 41,713
# Trainable params: 41,713
# Non-trainable params: 0