
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,SimpleRNN, Dropout
from tensorflow.keras.layers import Bidirectional, LSTM
#1.데이터구성
data = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6,],[5,6,7], [6,7,8],[7,8,9]])
y = np.array([4,5,6,7,8,9,10])

# x_shape(행, 열, 몇개씩 자르는지)
print(x.shape, y.shape)
x = x.reshape(7,3,1)
print(x.shape, y.shape)

model = Sequential()
model.add(Bidirectional(LSTM(64, return_sequences=True, input_shape=(3,1))))
model.add(LSTM(32,input_shape=(3,1), return_sequences=False))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.9))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))
# model.summary()

model.compile(loss = 'mse', optimizer= 'adam')
model.fit(x,y, epochs=100
          )

# 평가예측
print(model.evaluate(x,y))
y_pred = np.array([8,9,10]).reshape(1,3,1)
print(y_pred)
result = model.predict(y_pred)
print(result) # [[[8]],[9],[10]]]
# np array타입인 8910을 1,3,1로 리쉐잎하겠다
