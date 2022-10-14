import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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
model = Sequential()
# model.add(Dense(5, input_dim=3))
model.add(Dense(10, input_jl=(3,)))
model.add(Dense(5))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(5))
model.add(Dense(1))
model.summary()