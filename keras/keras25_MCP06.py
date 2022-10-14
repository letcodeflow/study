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


from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime as dt
date = dt.datetime.now()
date = date.strftime('%m%d %H%m')
filename = '{epoch:04d} {val_loss:.4f}.hdf5'
filepath = './_ModelCheckPoint/k25/6/'
MCP = ModelCheckpoint(monitor='val_loss', mode='auto', save_best_only=True, filepath=''.join([filepath,date,' ',filename]))
ES = EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

# scaler = MaxAbsScaler()
scaler = RobustScaler()
print(np.min(x_train))
print(np.max(x_train))
print(np.min(x_test))
print(np.max(x_test))

# robust
# 0.14
# 1680.0
# 0.13
# 1515.0

# model = Sequential()
# model.add(Dense(5, input_dim=13))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(10, activation='sigmoid'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(3, activation='softmax'))
# model.summary()
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 5)                 70
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
# dense_6 (Dense)              (None, 3)                 33
# =================================================================
# Total params: 3,933
# Trainable params: 3,933
# Non-trainable params: 0
# _______________________
input1 = Input(shape=(13,))
dense1 = Dense(5)(input1)
dense2 = Dense(100, activation='relu')(dense1)
dense3 = Dense(10, activation='relu')(dense2)
dense4 = Dense(100, activation='relu')(dense3)
dense5 = Dense(10, activation='sigmoid')(dense4)
dense6 = Dense(10, activation='relu')(dense5)
output1 = Dense(3, activation='softmax')(dense6)
model = Model(inputs=input1, outputs=output1)
model.summary()
# Model: "model"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_1 (InputLayer)         [(None, 13)]              0
# _________________________________________________________________
# dense (Dense)                (None, 5)                 70
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
# dense_6 (Dense)              (None, 3)                 33
# =================================================================
# Total params: 3,933
# Trainable params: 3,933
# Non-trainable params: 0
#3.컴파일 훈련
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(x_train, y_train, epochs=100, verbose=1, 
          validation_split=0.2,
          callbacks=[ES,MCP])

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

# maxabs
# loss :  0.7186919450759888
# acc : 0.6481481194496155
# acc 0.6481481481481481

# robust
# loss :  0.7514224052429199
# acc : 0.6666666865348816
# acc 0.6666666666666666