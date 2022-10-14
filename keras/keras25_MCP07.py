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
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler #대문자 클래스 약어도 ㄷ대문자로

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train))
print(np.max(x_train))
print(np.min(x_test))
print(np.max(x_test))
# maxabs
# 0.0
# 1.0
# 0.0
# 1.3333333333333333

# robust
# -2.6
# 16.0
# -2.6
# 16.0
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime as dt
date = dt.datetime.now()
date = date.strftime('%m%d %H%M')
filename = '{epoch:04d} {val_loss:.4f}.hdf5'
filepath = './_ModelCheckPoint/k25/7/'
mcp = ModelCheckpoint(filepath = ''.join([filepath, date,' ', filename]), save_best_only=True, monitor='val_loss', mode='auto')
ES = EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True)
# model = Sequential()
# model.add(Dense(5, input_dim=64))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(10, activation='sigmoid'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(10, activation='softmax'))
# model.summary()
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 5)                 325
# _________________________________________________________________
# dense_1 (Dense)              (None, 100)               600
# _________________________________________________________________
# dense_2 (Dense)              (None, 10)                1010
# _________________________________________________________________
# dense_3 (Dense)              (None, 100)               1100
# _________________________________________________________________
# dense_4 (Dense)              (None, 10)                1010
# _________________________________________________________________
# dense_5 (Dense)              (None, 10)                110
# _________________________________________________________________
# dense_6 (Dense)              (None, 10)                110
# =================================================================
# Total params: 4,265
# Trainable params: 4,265
# Non-trainable params: 0
input1 = Input(shape=(64,))
dense1 = Dense(5)(input1)
dense2 = Dense(100, activation='relu')(dense1)
dense3 = Dense(10, activation='relu')(dense2)
dense4 = Dense(100, activation='relu')(dense3)
dense5 = Dense(10, activation='sigmoid')(dense4)
dense6 = Dense(10, activation='relu')(dense5)
output1 = Dense(10, activation='softmax')(dense6)
model = Model(inputs=input1, outputs=output1)
model.summary()
# Model: "model"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_1 (InputLayer)         [(None, 64)]              0
# _________________________________________________________________
# dense (Dense)                (None, 5)                 325
# _________________________________________________________________
# dense_1 (Dense)              (None, 100)               600
# _________________________________________________________________
# dense_2 (Dense)              (None, 10)                1010
# _________________________________________________________________
# dense_3 (Dense)              (None, 100)               1100
# _________________________________________________________________
# dense_4 (Dense)              (None, 10)                1010
# _________________________________________________________________
# dense_5 (Dense)              (None, 10)                110
# _________________________________________________________________
# dense_6 (Dense)              (None, 10)                110
# =================================================================
# Total params: 4,265
# Trainable params: 4,265
# Non-trainable params: 0

#3.컴파일 훈련
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(x_train, y_train, epochs=100, verbose=1, 
          validation_split=0.2,
          callbacks=[ES,mcp])

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

# maxabs
# loss :  0.5009627938270569
# acc : 0.8870370388031006
# acc 0.8870370370370371

# robust
# loss :  0.403371125459671
# acc : 0.9111111164093018
# acc 0.9111111111111111