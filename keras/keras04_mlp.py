import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터

x = np.array([[1,2,3,4,5,6,7,8,9,10], 
             [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4]]
            )
y = np.array([11,12,13,14,15,16,17,18,19,20]
             )

print(x.shape)
print(y.shape)

# (2, 10)
# (10,)
'''
x =x.reshape(10,2)
print(x)

x= x.transpose()
print(x)
'''
x = x.T
print(x.shape)
print(y.shape)
print(x)

# (10, 2)
# (10,)
# [[ 1.   1. ]
#  [ 2.   1. ]
#  [ 3.   1. ]
#  [ 4.   1. ]
#  [ 5.   2. ]
#  [ 6.   1.3]
#  [ 7.   1.4]
#  [ 8.   1.5]
#  [ 9.   1.6]
#  [10.   1.4]]


#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=2))
model.add(Dense(4))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=500, batch_size=3)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss값은 ', loss)

result = model.predict([[10, 1.4]])
print('[10, 1.4]의 예측값은 ', result)

# loss값은  9.767093160917284e-07
# [10, 1.4]의 예측값은  [[20.000717]]

