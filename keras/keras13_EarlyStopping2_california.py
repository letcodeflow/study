#1.데이터
from gc import callbacks
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
model.add(Dense(200))
model.add(Dense(20))
model.add(Dense(200))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(200))
model.add(Dense(20))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss = 'mse', optimizer='adam')
from tensorflow.python.keras.callbacks import EarlyStopping
ES = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)
t=model.fit(x_train, y_train, epochs=650, batch_size=200, validation_split=0.01, callbacks=[ES])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss', loss)

# print('loss', loss)

# print('=======================================')
# print(t) #keras.callbacks.History object at 0x000001E3FA603A60
# print('=======================================')
# # print(hist.history)
# print('=======================================')
# # print(hist.history['loss'])
# print('=======================================')
# # print(hist.history['val_loss'])

# print(end_time - start_time)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2', r2)

import matplotlib.pyplot as plt

from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

plt.figure(figsize=(9,6)) #칸만들기
plt.plot(t.history['loss'], marker=',', c='red', label='loss')
#그림그릴거야
plt.plot(t.history['val_loss'], marker=',', c='blue', label='val_loss')
plt.grid()
plt.title('우결')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right') #라벨값위치 생략시 빈자리에 생성
plt.show()

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

# #ealy and hist
# loss 0.7296079993247986
# r2 0.4682816494450349

# loss 0.6573861241340637
# 194/194 [==============================] - 0s 615us/step
# r2 0.5209150257377738