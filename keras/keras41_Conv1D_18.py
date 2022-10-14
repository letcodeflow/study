import tensorflow as tf
tf.random.set_seed(137)
import numpy as np
from sklearn.datasets  import load_digits

#1.데이터
datasets = load_digits()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)
print(np.unique(y, return_counts=True))
# (1797, 64) (1797,)
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))
# from sklearn.preprocessing import OneHotEncoder
# encoder = OneHotEncoder()
# encoder.fit(y)
# y = encoder.transform(y).toarray()
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, shuffle=True, random_state=137)
from sklearn.preprocessing import MinMaxScaler, StandardScaler #대문자 클래스 약어도 ㄷ대문자로

# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train))
print(np.max(x_train))
print(np.min(x_test))
print(np.max(x_test))
x_train=x_train.reshape(-1,32,2)
x_test=x_test.reshape(-1,32,2)
from tensorflow.python.keras.layers import Dense, Dropout, LSTM, Conv1D, Flatten, MaxPool1D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping
ES = EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True)
model = Sequential()
model.add(Conv1D(16,2, input_shape=(32,2)))
model.add(MaxPool1D())
model.add(Flatten())
model.add(Dense(100, activation='relu',input_shape=(2,8,10)))
model.add(Dropout(0.3))
model.add(Dense(10, activation='relu',input_shape=(2,8,10)))
model.add(Dropout(0.3))
model.add(Dense(100, activation='relu',input_shape=(2,8,4)))
model.add(Dense(10, activation='sigmoid',input_shape=(2,8,4)))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu',input_shape=(2,8,4)))
model.add(Dense(10, activation='softmax',input_shape=(2,8,4)))
model.summary()
#3.컴파일 훈련
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(x_train, y_train, epochs=100, verbose=1, 
          validation_split=0.2,
          callbacks=ES)

#4.평가 예측
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)
# y_predict = model.predict(x_test[:5])
print(y_test)
print(y_predict)
y_predict = np.argmax(y_predict, axis= 1)
y_test = np.argmax(y_test, axis= 1)
print(y_test)
print(y_predict)

from sklearn.metrics import accuracy_score


# y_predict = y_predict.round(0)
# # pre2 = y_predict.flatten() # 차원 펴주기
# # pre3 = np.where(pre2 > 1, 2 , 0) #1보다크면 2, 작으면 0
acc = accuracy_score(y_test, y_predict)

# print(y_predict)
print('loss : ', loss[0])
#loss식의 첫번째
print('acc :',  loss[1])
#loss식의 두번째
print('acc', acc)

import matplotlib.pyplot as plt
plt.gray()
#흑백으로 그리겠다
plt.matshow(datasets.images[1])

plt.show()
print(datasets)
#미적용
# loss :  0.45412346720695496
# acc : 0.9092592597007751
# acc 0.9092592592592592

#민맥스
# loss :  0.49848970770835876
# acc : 0.8833333253860474
# acc 0.8833333333333333

#스탠
# loss :  0.4420080780982971
# acc : 0.8833333253860474
# acc 0.8833333333333333

# dropout
# loss :  0.9519795775413513
# acc : 0.5592592358589172
# acc 0.5592592592592592

# lstm
# loss :  1.592686414718628
# acc : 0.3092592656612396
# acc 0.30925925925925923

# maxpool1d conv1d
# loss :  0.6533721089363098
# acc : 0.8074073791503906
# acc 0.8074074074074075