import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import time

#1.데이터
x = np.array(range(1,17))
y = np.array(range(1,17))

#섞여있는 데이터셋 train:val:test = 10:3:3



from sklearn.model_selection import train_test_split

x_train, x_rem, y_train, y_rem = train_test_split(x,y, train_size=0.65, random_state=66)

x_val, x_test, y_val, y_test = train_test_split(x_rem, y_rem, train_size =0.5, random_state=66)
print(x_test)
print(x_val)
print(x_train)
print(x_test)
print(x_val)
print(x_train)
print(x_test)
print(x_val)
print(x_train)
print(x_test)
print(x_val)
print(x_train)
print(x_test)
print(x_val)
print(x_train)
print(x_test)
print(x_val)
print(x_train)
print(x_test)
print(x_val)
print(x_train)
print(x_test)
print(x_val)
print(x_train)
print(x_test)
print(x_val)
print(x_train)
print(x_test)
print(x_val)
print(x_train)
print(x_test)
print(x_val)
print(x_train)
print(x_test)
print(x_val)
print(x_train)
print(x_test)
print(x_val)
print(x_train)
print(x_test)
print(x_val)
print(x_train)
# # Let's say we want to split the data in 80:10:10 for train:valid:test dataset
# train_size=0.8

# # In the first step we will split the data in training and remaining dataset
# X_train, X_rem, y_train, y_rem = train_test_split(X,y, train_size=0.8)

# # Now since we want the valid and test size to be equal (10% each of overall data). 
# # we have to define valid_size=0.5 (that is 50% of remaining data)
# test_size = 0.5
# X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5)

print(x_train.shape), print(y_train.shape)
print(x_val.shape), print(y_val.shape)
print(x_test.shape), print(y_test.shape)


# x_train = x[:11]
# y_train = y[:11]
# x_test = x[11:14]
# y_test = y[11:14]
# x_val = x[14:17]
# y_val = y[14:17]
# x_train = np.array(range(1,11))
# y_train = np.array(range(1,11))
# x_test = np.array([11,12,13])
# y_test = np.array([11,12,13])
# x_val = np.array([14,15,16])
# y_val = np.array([14,15,16])



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
          validation_split=0.25)
end_time = time.time()
print('time', end_time-start_time)

#4.평가 예측
loss = model.evaluate(x_test, y_test)
print('loss', loss)
result = model.predict([17])
print('17 result', result)
