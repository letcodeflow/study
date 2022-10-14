#1. 데이터

import numpy as np

x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
from tensorflow.python.keras.models import Sequential
##이렇게 세부적인 기능만 불러오는건 텐서플로 전체를 불러오기엔 성능에 부담이 되기 때문인가?
from tensorflow.python.keras.layers import Dense

model = Sequential()
model.add(Dense(4, input_dim=1))
model.add(Dense(5))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([83])
print('4예측값은 ', result)

# loss :  3.789561370324927e-14
# 4예측값은  [[3.9999993]]