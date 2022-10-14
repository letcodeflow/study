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
hist = model.fit(x_train, y_train, epochs=250, batch_size=200, validation_split=0.1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss', loss)

import matplotlib.pyplot as plt

from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

plt.figure(figsize=(10,8))
plt.plot(model.fit(x_train, y_train, epochs=250, batch_size=200, validation_split=0.1).history['loss'], marker='.', c='red', label='loss')
plt.plot(model.fit(x_train, y_train, epochs=250, batch_size=200, validation_split=0.1).history['val_loss'], marker='_',c='blue', label='val_loss')
plt.grid()
plt.title('합쳐')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()
plt.show()

