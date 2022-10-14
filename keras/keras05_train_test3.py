import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])



#train과 test를 섞어 같은 특성을 지니게 한다

x_train, x_test, y_train, y_test = train_test_split(
    x, y, #test_size=0.3, 
    train_size=0.7,
    #shuffle=False,
    random_state=66
)

print(x_train)
print(x_test)
print(y_train)
print(y_test)

# [2 7 6 3 4 8 5]
# [ 1  9 10]
# [2 7 6 3 4 8 5]
# [ 1  9 10]

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs =100, batch_size=1)

#4 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss ', loss)

result = model.predict([11])
print('11의 예측값', result)

# loss  8.431605238001794e-06
# 11의 예측값 [[10.996157]]

