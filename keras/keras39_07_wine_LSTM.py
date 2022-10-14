import tensorflow as tf
tf.random.set_seed(137)
import numpy as np
from sklearn.datasets  import load_wine

#1.데이터
datasets = load_wine()
x = datasets.data
y = datasets.target.reshape(-1,1)
print(x.shape, y.shape)
print(np.unique(y, return_counts=True))
# (array([0, 1, 2]), array([59, 71, 48], dtype=int64))

print(datasets.DESCR)

    # :Number of Instances: 178 (50 in each of three classes)
    # :Number of Attributes: 13 numeric,
print(datasets.feature_names)
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
encoder.fit(y)
y = encoder.transform(y).toarray()
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, shuffle=True, random_state=137)

x_train=x_train.reshape(-1,1,13)
x_test=x_test.reshape(-1,1,13)
from tensorflow.python.keras.layers import Dense, Dropout, LSTM
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping
ES = EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True)
model = Sequential()
model.add(LSTM(5, input_shape=(1,13)))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))

#3.컴파일 훈련
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(x_train, y_train, epochs=1, verbose=1, 
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

#민맥스
# loss :  0.24404345452785492
# acc : 0.9629629850387573
# acc 0.9629629629629629

#스탠
# loss :  0.12898489832878113
# acc : 0.9629629850387573
# acc 0.9629629629629629

#스케일미적용
# loss :  0.7572556138038635
# acc : 0.6481481194496155
# acc 0.6481481481481481

# dropout
# loss :  1.0873541831970215
# acc : 0.29629629850387573
# acc 0.2962962962962963

# lstm
# loss :  1.091688632965088
# acc : 0.40740740299224854
# acc 0.4074074074074074