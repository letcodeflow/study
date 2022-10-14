import numpy as np


#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
           [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
           [9,8,7,6,5,4,3,2,1,0]]
           )
y = np.array([11,12,13,14,15,16,17,18,19,20])

print(x.shape, y.shape)
x = np.transpose(x)
print(x.shape)

#2.모델
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense,Input, Dropout
# model = Sequential()
# # model.add(Dense(5, input_dim=3))
# model.add(Dense(10, input_shape=(3,)))
# model.add(Dense(5))
# model.add(Dense(50))
# model.add(Dense(50))
# model.add(Dense(50))
# model.add(Dense(5))
# model.add(Dense(1))
input1 = Input(shape=(3,))
#이름변경 가능 나중에 input2,3 추가될 수도 있음
dense1 = Dense(10)(input1)
dense2 = Dense(5, activation='relu')(dense1)
dense3 = Dense(5, activation='sigmoid')(dense2)
drop1 = Dropout(0.2)(dense3)
dense4 = Dense(5)(drop1)
outpit1 = Dense(1)(dense4)
model = Model(inputs=input1, outputs=outpit1)
#레이어 재사용
model.summary()
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 10)                40
# _________________________________________________________________
# dense_1 (Dense)              (None, 5)                 55
# _________________________________________________________________
# dense_2 (Dense)              (None, 50)                300
# _________________________________________________________________
# dense_3 (Dense)              (None, 50)                2550
# _________________________________________________________________
# dense_4 (Dense)              (None, 50)                2550
# _________________________________________________________________
# dense_5 (Dense)              (None, 5)                 255
# _________________________________________________________________
# dense_6 (Dense)              (None, 1)                 6
# =================================================================
# Total params: 5,756
# Trainable params: 5,756
# Non-trainable params: 0

# Model: "model"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_1 (InputLayer)         [(None, 3)]               0
# _________________________________________________________________
# dense (Dense)                (None, 10)                40
# _________________________________________________________________
# dense_1 (Dense)              (None, 5)                 55
# _________________________________________________________________
# dense_2 (Dense)              (None, 5)                 30
# _________________________________________________________________
# dense_3 (Dense)              (None, 5)                 30
# _________________________________________________________________
# dense_4 (Dense)              (None, 1)                 6
# =================================================================
# Total params: 161
# Trainable params: 161
# Non-trainable params: 0

#
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x,y, epochs=100, batch_size=1)
