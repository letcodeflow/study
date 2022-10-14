#1.데이터
from sklearn.datasets import fetch_california_housing
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

print(x.shape)
print(y.shape)

print(datasets.feature_names)
print(datasets.DESCR)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=66)

#2.모델

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model=Sequential()
model.add(Dense(20, input_dim=8))
model.add(Dense(80))
model.add(Dense(20))
model.add(Dense(80))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=250, batch_size=200, validation_split=0.01)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2=r2_score(y_test, y_predict)
print('r2', r2)

#validation 이전

# loss 0.6581024527549744
# 194/194 [==============================] - 0s 664us/step
# r2 0.5203931464750856

# loss 0.6531566381454468
# 194/194 [==============================] - 0s 674us/step
# r2 0.5239972327558571

# loss 0.650373637676239
# 194/194 [==============================] - 0s 639us/step
# r2 0.5260254847020475

#적용후
# loss 0.6490857005119324
# 194/194 [==============================] - 0s 615us/step
# r2 0.5269640664307653

# loss 0.6621416807174683
# 194/194 [==============================] - 0s 632us/step
# r2 0.5174494649013497

# loss 0.7691570520401001
# 194/194 [==============================] - 0s 654us/step
# r2 0.4394595104425515