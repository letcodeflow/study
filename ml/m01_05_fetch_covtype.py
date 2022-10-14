import tensorflow as tf
tf.random.set_seed(137)
import numpy as np
from sklearn.datasets import fetch_covtype
#1.데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
# print(y)
# print(x.shape, y.shape)
# print(np.unique(y, return_counts=True))
# (581012, 54) (581012,)
# (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],
#       dtype=int64))
# from sklearn.preprocessing import OneHotEncoder
# encoder = OneHotEncoder()
# encoder.fit(y)
# y = encoder.transform(y).toarray()

# print(y)
# print(x.shape, y.shape)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, shuffle=True, random_state=137)

from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.svm import LinearSVC
ES = EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True)
# model = Sequential()
# model.add(Dense(5, input_dim=54))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(10, activation='sigmoid'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(7, activation='softmax'))
model = LinearSVC()
model.fit(x_train, y_train)
#3.컴파일 훈련
# model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# hist = model.fit(x_train, y_train, epochs=10, verbose=1, 
#           validation_split=0.2,
#           callbacks=ES)


#4.평가 예측
# loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)
# y_predict = model.predict(x_test[:5])
print(y_test)
print(y_predict)
# y_predict = np.argmax(y_predict, axis= 1)
# y_test = np.argmax(y_test, axis= 1)
print(y_test)
print(y_predict)

from sklearn.metrics import accuracy_score


# y_predict = y_predict.round(0)
# # pre2 = y_predict.flatten() # 차원 펴주기
# # pre3 = np.where(pre2 > 1, 2 , 0) #1보다크면 2, 작으면 0
acc = accuracy_score(y_test, y_predict)

# print(y_predict)
# print('loss : ', loss[0])
#loss식의 첫번째
# print('acc :',  loss[1])
#loss식의 두번째
print('acc', acc)
print(model.score(x_test, y_test))