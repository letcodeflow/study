import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1.데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])

#트레인 데이터셋과 테스트 평가셋 나누기

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, test_size=0.3, shuffle=True, random_state=66)

print(x_train)
print(x_test)
print(y_train)
print(y_test)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x, y, epochs = 100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print(loss)
print(x)
y_predict = model.predict(x) #x를 넣었을때 y가 얼마나오느냐
print(y_predict)

from sklearn.metrics import r2_score
r2 = r2_score(y, y_predict)
print(r2)