#1. 데이터
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import time


pd.__version__
path = './_data/ddareunge/'
train_set = pd.read_csv(path + 'train.csv', index_col=0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)

print(test_set.shape)

#자료구조를 봐야하니까
print(train_set)
print(train_set.shape)
print(train_set.describe())
print(train_set.info())

print('널값이 ', train_set.isnull().sum())
train_set = train_set.dropna()
print('널값이 삭제이후 ', train_set.isnull().sum())
print(train_set.shape)


x = train_set.drop(['count'], axis=1)
print('x컬럼', x.shape)
y = train_set['count']
print('y 컬럼', y.shape)


x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.99, shuffle=False)
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
# 1.0
# robust
# -1.9024390243902443
# 6.382352941176471
# -1.1883662803908204
# 2.5294117647058822
print(x_train)
print(y_train)

#2.모델구성

# model = Sequential()
# model.add(Dense(200, input_dim=9))
# model.add(Dense(200))
# model.add(Dense(200))
# model.add(Dense(200))
# model.add(Dense(200))
# model.add(Dense(200))
# model.add(Dense(200))
# model.add(Dense(200))
# model.add(Dense(1))
# model.summary(0)

# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  dense (Dense)               (None, 200)               2000

#  dense_1 (Dense)             (None, 200)               40200

#  dense_2 (Dense)             (None, 200)               40200

#  dense_3 (Dense)             (None, 200)               40200

#  dense_4 (Dense)             (None, 200)               40200

#  dense_5 (Dense)             (None, 200)               40200

#  dense_6 (Dense)             (None, 200)               40200

#  dense_7 (Dense)             (None, 200)               40200

#  dense_8 (Dense)             (None, 1)                 201

# =================================================================
# Total params: 283,601
# Trainable params: 283,601
# Non-trainable params: 0

input1 = Input(shape=(9,))
dense1 = Dense(200)(input1)
dense2 = Dense(200)(dense1)
dense3 = Dense(200)(dense2)
dense4 = Dense(200)(dense3)
dense5 = Dense(200)(dense4)
dense6 = Dense(200)(dense5)
dense7 = Dense(200)(dense6)
dense8 = Dense(200)(dense7)
output1 = Dense(1)(dense8)
model = Model(inputs=input1, outputs=output1)
model.summary(0)
# Model: "model"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  input_1 (InputLayer)        [(None, 9)]               0

#  dense (Dense)               (None, 200)               2000

#  dense_1 (Dense)             (None, 200)               40200

#  dense_2 (Dense)             (None, 200)               40200

#  dense_3 (Dense)             (None, 200)               40200

#  dense_4 (Dense)             (None, 200)               40200

#  dense_5 (Dense)             (None, 200)               40200

#  dense_6 (Dense)             (None, 200)               40200

#  dense_7 (Dense)             (None, 200)               40200

#  dense_8 (Dense)             (None, 1)                 201

# =================================================================
# Total params: 283,601
# Trainable params: 283,601
# Non-trainable params: 0

#3. 컴파일 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 45, batch_size=200, verbose=1)

start_time = time.time()

#4. 평가 및 예측
loss = model.evaluate(x_test, y_test)
print('loss', loss)

y_predict = model.predict(x_test)
# print(y_predict)

r2 = r2_score(y_test, y_predict)
print('r2', r2)

#rmse
def RMSE(a,b):
    return np.sqrt(mean_squared_error(a,b))

rmse = RMSE(y_test, y_predict)
print('rmse', rmse)

end_time = time.time() - start_time
print(end_time)

# loss 2116.8349609375
# 3/3 [==============================] - 0s 4ms/step
# r2 0.7308391739043492
# rmse 46.00907729565719

# loss 2223.976806640625
# 1/1 [==============================] - 0s 94ms/step
# r2 0.8147804410965995
# rmse 47.1590564329388
# 0.383059024810791


y_summit = model.predict(test_set)
print(y_summit)
print(y_summit.shape) #(715, 1)

## .to_csv() 로 submission.csv 에 입력

submission = pd.read_csv(path + 'submission.csv')#함수정의하고 값 불러오기
submission['count'] = y_summit #카운트에 y_summit 덮어씌우기


submission.to_csv(path + 'submission.csv') #y_summit이 덮어씌어진 submission을 불러온 파일에 다시 덮어씌우기
# loss 2489.98486328125
# r2 0.7926264243669512
# rmse 49.89975258447158
# 0.3418304920196533

#민맥스
# loss 2868.696533203125
# r2 0.7610862259980802
# rmse 53.5602109213241
# 0.33219313621520996

#스탠
# loss 2199.585693359375
# r2 0.8168118026787894
# rmse 46.899739313256795
# 0.3275487422943115
print('loss', loss)
print('r2', r2)
print('rmse', rmse)

# maxabas
# loss 2092.291015625
# r2 0.8257476250201532
# rmse 45.74156687777579

# robust
# loss 2472.82177734375
# r2 0.7940558715495929
# rmse 49.72747314715278