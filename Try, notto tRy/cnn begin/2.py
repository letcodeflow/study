from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler
#1.데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)
import numpy as np
import pandas as pd

print(y_train.unique())

""" 
print(np.unique(y_train, return_counts=True))
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

#    0  1  2  3  4  5  6  7  8  9
# 0  0  0  0  0  0  1  0  0  0  0
# 1  1  0  0  0  0  0  0  0  0  0
# 2  0  0  0  0  1  0  0  0  0  0
# 3  0  1  0  0  0  0  0  0  0  0
# 4  0  0  0  0  0  0  0  0  0  1
#    0  1  2  3  4  5  6  7  8  9
# 0  0  0  0  0  0  1  0  0  0  0
# 1  1  0  0  0  0  0  0  0  0  0
# 2  0  0  0  0  1  0  0  0  0  0
# 3  0  1  0  0  0  0  0  0  0  0
# 4  0  0  0  0  0  0  0  0  0  1
# (60000, 10) (10000, 10)


from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# [[[1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [0. 1.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]]

#  [[0. 1.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]]

#  [[1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [0. 1.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]]

#  [[1. 0.]
#   [0. 1.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]]

#  [[1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [0. 1.]]]
# [[[1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [0. 1.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]]

#  [[0. 1.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]]

#  [[1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [0. 1.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]]

#  [[1. 0.]
#   [0. 1.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]]

#  [[1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [1. 0.]
#   [0. 1.]]]
# (60000, 10, 2) (10000, 10, 2)

print(y_train[:5])
print(y_train[:5])
print(y_train.shape, y_test.shape)
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='valid', input_shape=(28,28,1)))
model.add(MaxPooling2D())
model.add(Conv2D(5,(2,2), padding='valid',activation='relu'))
model.add(Conv2D(7,(2,2), padding='valid', activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.summary()

#3.컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=200)

#4.평가 훈련
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print(y_test[:5])
print(y_predict[:5])
print(y_test.shape, y_predict.shape)
# y_predict = np.argmax(y_predict, axis=1)
# y_test = np.argmax(y_test, axis=1)

# from sklearn.metrics import accuracy_score
# acc = accuracy_score(y_test, y_predict)
# print(acc) """