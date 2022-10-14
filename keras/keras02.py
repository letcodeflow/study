import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#1. 데이터
x = np.array([1,2,3,5,4])
y = np.array([1,2,3,4,5])

#2. 모델구성
model = Sequential()
model.add(Dense(8, input_dim=1))
model.add(Dense(50))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(20))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=5000)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([6])
print('6예측값은 ', result)

# loss :  0.4042775630950928
# 6예측값은  [[5.9232044]]

# loss :  0.42744380235671997
# 6예측값은  [[6.035554]]