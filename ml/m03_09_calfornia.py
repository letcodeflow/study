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
from sklearn.svm import LinearSVR
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=66)
#스케일링팥
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
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

#2.모델
# x_train=x_train.reshape(-1,2,4)
# x_test=x_test.reshape(-1,2,4)
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, LSTM, Conv1D, MaxPool1D,Flatten
from sklearn.svm import LinearSVR, SVR
from sklearn.linear_model import Perceptron,LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

model = SVR()
model = LinearRegression()
model = KNeighborsRegressor()
model = DecisionTreeRegressor()
model = RandomForestRegressor()

# svr
# r2 -0.0370628287403556
# -0.0370628287403556

# linnearr
# r2 0.6001949284390904
# 0.6001949284390904

# kneighbor
# r2 0.13188588139050017
# 0.13188588139050017

# decsiociontree
# r2 0.6004119613306693
# 0.6004119613306693

# random
# r2 0.8095790408763213
# 0.8095790408763213

# model=Sequential()
# model.add(Conv1D(20,1, input_shape=(2,4))) 

# model.add(MaxPool1D())
# model.add(Dense(20))
# model.add(Dropout(0.3))
# model.add(Dense(80))
# model.add(Dense(20))
# model.add(Dropout(0.3))
# model.add(Dense(20))
# model.add(Dense(20))
# model.add(Dropout(0.3))
# model.add(Dense(20))
# model.add(Dense(20))
# model.add(Dense(20))
# model.add(Dropout(0.2))
# model.add(Dense(20))
# model.add(Dense(1))
# model.summary()

#3. 컴파일 훈련
# model.compile(loss = 'mse', optimizer='adam')
# model.fit(x_train, y_train, epochs=250, batch_size=200, validation_split=0.01)

#4. 평가, 예측
# loss = model.evaluate(x_test, y_test)
# print('loss', loss)
# model = LinearSVR()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
print(y_predict.shape, y_test.shape)
print(y_predict)
print(y_test)

# y_predict = y_predict.reshape(-1,1)
# y_test = y_test.reshape(-1,1)
from sklearn.metrics import r2_score
r2=r2_score(y_test, y_predict)
print('r2', r2)
print(model.score(x_test,y_test))
# linearsvr
# r2 -3.891853644131518
# -3.891853644131518
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

# Dropout
# loss 0.5472083687782288
# r2 0.60120947610524

# lstm
# loss 0.3066251277923584
# r2 0.7765400196338007

# maxpool1d conv1d
# loss 0.8805951476097107
# r2 0.3582462196326637