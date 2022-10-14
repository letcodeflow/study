#1. 데이터 불러오기

from socket import fromfd
from sklearn.datasets import load_diabetes
data_sets = load_diabetes()
x = data_sets.data
y = data_sets.target

#트레인값
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=False
)
print(x.shape, y.shape)
# (442, 10) (442,) dim=10
#이진모델인지 확인
print(data_sets.feature_names)
print(data_sets.DESCR)


#모델구성
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
model = Sequential()
model.add(Dense(5, input_dim=10))
model.add(Dense(5, activation='sigmoid'))
#activation을 덴스 인수 안에 써야함
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='sigmoid'))
model.add(Dense(5))
model.add(Dense(1))

#3.컴파일 훈련
model.compile(loss = 'mse', optimizer='adam', 
              metrics=['accuracy', 'mse'])
from tensorflow.python.keras.callbacks import EarlyStopping
ES = EarlyStopping(monitor='val_loss', patience=20, 
                   mode ='min', restore_best_weights=True)
histo = model.fit(x_train, y_train, epochs=100,
                  batch_size=20, 
                #   validation_split=0.2,
                callbacks=[ES],
                  verbose=2)

#4.평가 예측
loss = model.evaluate(x_test, y_test)
# r2스코어 구하기, 그림그리기
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)

print('loss', loss)
print('r2', r2)

#그리기
import matplotlib.pyplot as plt
plt.figure(figsize=(3,2))
plt.plot(histo.history['loss'], marker = '.', c='red', label='loss')
plt.plot(histo.history['val_loss'], marker = '.', c='blue', label='val_loss')
plt.xlabel('label')
plt.ylabel('y')
plt.grid()
plt.legend(loc='upper left')
plt.show()

# 적용
# loss [7295.38623046875, 0.0, 7295.38623046875]
# r2 -0.13603971387544767

# 미적용
# loss [6668.47314453125, 0.0, 6668.47314453125]
# r2 -0.03841684077450647

# 미적용일때 더 오류가 적었다. 리니어모델여부는 아닌것 같다