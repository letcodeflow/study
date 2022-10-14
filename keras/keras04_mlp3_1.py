import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([range(10), range(21, 31), range(201, 211)])
#print(x)
# [[  0   1   2   3   4   5   6   7   8   9]
#  [ 21  22  23  24  25  26  27  28  29  30]
#  [201 202 203 204 205 206 207 208 209 210]]

# for i in range(10): #
#     print(i)
print(x.shape) #(3, 10)
x= np.transpose(x)
print(x.shape) #(10, 3)

y = np.array([[1,2,3,4,5,6,7,8,9,10],
           [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]]
           )
y = np.transpose(y)
print(y.shape)

#2.모델
model = Sequential()
model.add(Dense(5, input_dim=3))
model.add(Dense(5))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(5))
model.add(Dense(2))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=500, batch_size = 2)

#4. 평가 예측
loss = model.evaluate(x, y)
print('loss 값은', loss)

result = model.predict([[9, 30, 201]])
print('[[9, 30, 201]]의 예측값은 ', result)

# loss 값은 1.1076099759588232e-10
# [[9, 30, 201]]의 예측값은  [[10.2242565  1.8520571]]








#예측값[[9, 30, 201]]
