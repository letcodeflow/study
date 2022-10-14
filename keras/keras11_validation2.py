import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import time

#1.데이터
x = np.array(range(0,16))
y = np.array(range(0,16))
print('0부터 16')
x_train = x[:11]
y_train = y[:11]
x_test = x[11:14]
y_test = y[11:14]
x_val = x[14:17]
y_val = y[14:17]
print(x_train)
print(y_train)
print(x_test)
print(y_test)
print(x_val)
print(x_val)

print('1부터 17')

x = np.array(range(1,17))
y = np.array(range(1,17))

x_train = x[:11]
y_train = y[:11]
x_test = x[11:14]
y_test = y[11:14]
x_val = x[14:]
y_val = y[14:]
print(x_train)
print(y_train)
print(x_test)
print(y_test)
print(x_val)
print(x_val)
# x_train = np.array(range(1,11))
# y_train = np.array(range(1,11))
# x_test = np.array([11,12,13])
# y_test = np.array([11,12,13])
# x_val = np.array([14,15,16])
# y_val = np.array([14,15,16])


'''
#2.모델

model = Sequential()
model.add(Dense(100, input_dim=1))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(1))

#3.컴파일 훈련
model.compile(loss = 'mse', optimizer = 'adam')
start_time = time.time()
model.fit(x_train, y_train, epochs = 100, verbose=1, batch_size=20,
          validation_data=(x_val, y_val))
end_time = time.time()
print('time', end_time-start_time)

#4.평가 예측
loss = model.evaluate(x_test, y_test)
print('loss', loss)
result = model.predict([17])
print('17 result', result)
'''