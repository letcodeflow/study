import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터

x = np.array([[1,2,3,4,5,6,7,8,9,10], 
             [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4],
             [9,8,7,6,5,4,3,2,1,0]]
            )
y = np.array([11,12,13,14,15,16,17,18,19,20]
             )

print(x.shape)
print(y.shape)


x = x.T
print(x.shape)
print(y.shape)
print(x)

#3. 모델구성
model=Sequential()
model.add(Dense(5, input_dim=3))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

# (10, 3)
# (10,)
# [[ 1.   1.   9. ]
#  [ 2.   1.   8. ]
#  [ 3.   1.   7. ]
#  [ 4.   1.   6. ]
#  [ 5.   2.   5. ]
#  [ 6.   1.3  4. ]
#  [ 7.   1.4  3. ]
#  [ 8.   1.5  2. ]
#  [ 9.   1.6  1. ]
#  [10.   1.4  0. ]]


#예측 : [[10, 1.4, 0]]

#4. 컴파일 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=500, batch_size=3)


#5. 평가 예측
loss = model.evaluate(x,y)
print('loss값', loss)
result = model.predict([[10, 1.4, 0]])
print('[[10, 1.4, 0]] 예측값은 ', result)

# loss값 0.45124444365501404
# [[10, 1.4, 0]] 예측값은  [[19.025507]]
