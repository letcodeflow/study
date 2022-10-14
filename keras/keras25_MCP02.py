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
#스케일링팥
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, maxabs_scale
import numpy as np
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
#minmax
# 0.0
# 1.0
# -0.0050864699898269805
# 2.0745532963647566
#stan
# -2.369187755757429
# 109.92119256376915
# -2.3942132659109805
# 228.48436764973988
# MaxAbs
# -1.0
# 1.0
# -1.0004022526146419
# 2.0732094648245196
# robust
# -7.639321350109111
# 692.7733530866274
# -5.859629560171584
# 1439.7772528385135
#2.모델

from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime as dt
date = dt.datetime.now()
date = date.strftime('%m%d-%H%M')
filename = '{epoch:05d}_{val_loss}.hdf5'
MCP = ModelCheckpoint(monitor='val_loss', mode='auto', save_best_only=True, filepath=''.join(['./_ModelCheckPoint/k25/2/',date,'_', filename]))
# model=Sequential()
# model.add(Dense(20, input_dim=8))
# model.add(Dense(80))
# model.add(Dense(20))
# model.add(Dense(80))
# model.add(Dense(20))
# model.add(Dense(20))
# model.add(Dense(20))
# model.add(Dense(20))
# model.add(Dense(20))
# model.add(Dense(20))
# model.add(Dense(20))
# model.add(Dense(1))
# model.summary()

# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 20)                180
# _________________________________________________________________
# dense_1 (Dense)              (None, 80)                1680
# _________________________________________________________________
# dense_2 (Dense)              (None, 20)                1620
# _________________________________________________________________
# dense_3 (Dense)              (None, 80)                1680
# _________________________________________________________________
# dense_4 (Dense)              (None, 20)                1620
# _________________________________________________________________
# dense_5 (Dense)              (None, 20)                420
# _________________________________________________________________
# dense_6 (Dense)              (None, 20)                420
# _________________________________________________________________
# dense_7 (Dense)              (None, 20)                420
# _________________________________________________________________
# dense_8 (Dense)              (None, 20)                420
# _________________________________________________________________
# dense_9 (Dense)              (None, 20)                420
# _________________________________________________________________
# dense_10 (Dense)             (None, 20)                420
# _________________________________________________________________
# dense_11 (Dense)             (None, 1)                 21
# =================================================================
# Total params: 9,321
# Trainable params: 9,321
# Non-trainable params: 0

input1 = Input(shape=(8,))
dense1 = Dense(20)(input1)
dense2 = Dense(80)(dense1)
dense3 = Dense(20)(dense2)
dense4 = Dense(80)(dense3)
dense5 = Dense(20)(dense4)
dense6 = Dense(20)(dense5)
dense7 = Dense(20)(dense6)
dense8 = Dense(20)(dense7)
dense9 = Dense(20)(dense8)
dense10 = Dense(20)(dense9)
dense11 = Dense(20)(dense10)
output1 = Dense(1)(dense11)
model = Model(inputs=input1, outputs=output1)
model.summary()
# Model: "model"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_1 (InputLayer)         [(None, 8)]               0
# _________________________________________________________________
# dense (Dense)                (None, 20)                180
# _________________________________________________________________
# dense_1 (Dense)              (None, 80)                1680
# _________________________________________________________________
# dense_2 (Dense)              (None, 20)                1620
# _________________________________________________________________
# dense_3 (Dense)              (None, 80)                1680
# _________________________________________________________________
# dense_4 (Dense)              (None, 20)                1620
# _________________________________________________________________
# dense_5 (Dense)              (None, 20)                420
# _________________________________________________________________
# dense_6 (Dense)              (None, 20)                420
# _________________________________________________________________
# dense_7 (Dense)              (None, 20)                420
# _________________________________________________________________
# dense_8 (Dense)              (None, 20)                420
# _________________________________________________________________
# dense_9 (Dense)              (None, 20)                420
# _________________________________________________________________
# dense_10 (Dense)             (None, 20)                420
# _________________________________________________________________
# dense_11 (Dense)             (None, 1)                 21
# =================================================================
# Total params: 9,321
# Trainable params: 9,321
# Non-trainable params: 0

#3. 컴파일 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=250, batch_size=200, validation_split=0.01, callbacks=MCP)

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

#minmax
# loss 0.5488771796226501
# r2 0.5999933877663844

#stan
# loss 0.5570531487464905
# r2 0.594034815118113

#조금 높아졌음

# maxabs_scal
# loss 0.5944863557815552
# r2 0.5667545287993054

# robust
# loss 0.5443264245986938
# r2 0.6033097620311498