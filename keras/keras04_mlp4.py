import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([range(10)])
x= np.transpose(x)
print(x.shape) #(10, 1)

y = np.array([[1,2,3,4,5,6,7,8,9,10],
           [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
           [9,8,7,6,5,4,3,2,1,0]]
           )
y = np.transpose(y)
print(y.shape)

#2.모델
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(5))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(5))
model.add(Dense(3))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=500, batch_size = 2)

#4. 평가 예측
loss = model.evaluate(x, y)
print('loss 값은', loss)

result = model.predict([[9]])
print('[9]의 예측값은 ', result)

# loss 값은 0.0023409524001181126
# [[9, 30, 201]]의 예측값은  [[10.031815    1.9597656   0.12863034]]






#예측값[[9, 30, 201]]
