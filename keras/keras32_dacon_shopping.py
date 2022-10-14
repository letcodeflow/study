# 1. train.csv : 학습 데이터
# id : 샘플 아이디
# Store : 쇼핑몰 지점
# Date : 주 단위(Weekly) 날짜
# Temperature : 해당 쇼핑몰 주변 기온
# Fuel_Price : 해당 쇼핑몰 주변 연료 가격
# Promotion 1~5 : 해당 쇼핑몰의 비식별화된 프로모션 정보
# Unemployment : 해당 쇼핑몰 지역의 실업률
# IsHoliday : 해당 기간의 공휴일 포함 여부
# Weekly_Sales : 주간 매출액 (목표 예측값)


# 2. test.csv : 테스트 데이터
# id : 샘플 아이디
# Store : 쇼핑몰 지점
# Date : 주 단위(Weekly) 날짜
# Temperature : 해당 쇼핑몰 주변 기온
# Fuel_Price : 해당 쇼핑몰 주변 연료 가격
# Promotion 1~5 : 해당 쇼핑몰의 비식별화된 프로모션 정보
# Unemployment : 해당 쇼핑몰 지역의 실업률
# IsHoliday : 해당 기간의 공휴일 포함 여부


# 3. sample_submission.csv : 제출 양식
# id : 샘플 아이디
# Weekly_Sales : 주간 매출액 (목표 예측값)

path = 'c:/study/_data/dacon_shop/'
import pandas as pd
train_set = pd.read_csv(path +'train.csv', index_col=0)
test_set = pd.read_csv(path +'test.csv', index_col=0)
submission = pd.read_csv(path +'sample_submission.csv', index_col=0)

# print(train_set.columns)
# 'id', 'Store', 'Date', 'Temperature', 'Fuel_Price', 'Promotion1',
#        'Promotion2', 'Promotion3', 'Promotion4', 'Promotion5', 'Unemployment',
#        'IsHoliday', 'Weekly_Sales
# print(test_set.columns)
# id', 'Store', 'Date', 'Temperature', 'Fuel_Price', 'Promotion1',
#        'Promotion2', 'Promotion3', 'Promotion4', 'Promotion5', 'Unemployment',
#        'IsHoliday']
# print(submission.columns)
# 'id', 'Weekly_Sales'
#답안지 submission['Weekly_Sales'] = y_summit
#id는 다 날림 결측치와 문자컬럼값 확인
# print(train_set.isnull().sum())
# print(test_set.isnull().sum())
# Store              0
# Date               0
# Temperature        0
# Fuel_Price         0
# Promotion1      4153
# Promotion2      4663
# Promotion3      4370
# Promotion4      4436
# Promotion5      4140
# Unemployment       0
# IsHoliday          0
# Weekly_Sales       0
# dtype: int64
# Store             0
# Date              0
# Temperature       0
# Fuel_Price        0
# Promotion1        2
# Promotion2      135
# Promotion3       19
# Promotion4       34
# Promotion5        0
# Unemployment      0
# IsHoliday         0
# dtype: int64
#프로모션 1~5 결측치 확인 남은 값이 아마 중요한듯
#고유값확인 테스트의 프로모션 1,2,3,4는 결측치가 적으므로 해당 커럼 평균으로 챙우기
test_set['Promotion1'] = test_set['Promotion1'].fillna(test_set['Promotion1'].mean())
test_set['Promotion2'] = test_set['Promotion2'].fillna(test_set['Promotion2'].mean())
test_set['Promotion3'] = test_set['Promotion3'].fillna(test_set['Promotion3'].mean())
test_set['Promotion4'] = test_set['Promotion4'].fillna(test_set['Promotion4'].mean())
test_set['Promotion5'] = test_set['Promotion5'].fillna(test_set['Promotion5'].mean())
train_set['Promotion1'] = train_set['Promotion1'].fillna(test_set['Promotion1'].mean())
train_set['Promotion2'] = train_set['Promotion2'].fillna(test_set['Promotion2'].mean())
train_set['Promotion3'] = train_set['Promotion3'].fillna(test_set['Promotion3'].mean())
train_set['Promotion4'] = train_set['Promotion4'].fillna(test_set['Promotion4'].mean())
train_set['Promotion5'] = train_set['Promotion5'].fillna(test_set['Promotion4'].mean())
print(test_set.isnull().sum()) #cool
print(train_set.isnull().sum()) #cool
# train_set = pd.get_dummies(train_set, columns=['IsHoliday'])
# test_set = pd.get_dummies(test_set, columns=['IsHoliday'])

# test_set['IsHoliday_True'] = 0
# print(train_set.shape, test_set.shape) (6255, 12) (180, 11)
#트레인셋의 프로모션1~4가 결측치가 많지만 테스트셋에서 버릴수 없으므로 테스트셋평균으로 채워줌
#date, promotion5 삭제 홀리데이 원핫인코딩필요 언임플로이먼트수치데이터로 여겨짐 난값 채우고 스케일링 필요할듯
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import StandardScaler,OneHotEncoder
# ct = ColumnTransformer( [("scaling", StandardScaler(), ['IsHoliday']),  ("onehot", OneHotEncoder(sparse = False), ['IsHoliday'])])
# train_set2=ct.fit_transform(train_set)
# print(train_set.columns)
print(test_set)
print(train_set)

print(train_set.info())
print(test_set.info())
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# ct = ColumnTransformer([("onehot",OneHotEncoder(sparse = False), ['IsHoliday'])]
# )
# ct.fit_transform(train_set)
# ct.fit_transform(test_set)


# print(train_set.columns)
# print(test_set.columns)
# one_hot_encoder.categories_

import numpy as np
#xy나누고 언임플로이 스케일링
#프로모션5 테스트셋은 꽉차있기때문에 해당부분으로 트레인셋채움
x = train_set.drop(['Store', 'Date','Weekly_Sales','IsHoliday','Promotion1','Promotion2','Promotion3','Promotion4'], axis=1)
#test_set도 같이 처리 대신 답안열은 제거 서브미션에 붙어있으므로
test_set = test_set.drop(['Store', 'Date','IsHoliday','Promotion1','Promotion2','Promotion3','Promotion4'], axis=1)
y= train_set['Weekly_Sales']
#와꾸확인
print(test_set.shape, submission.shape) #(180, 8) (180, 1)
print(x.shape, y.shape)# (6255, 9) (6255,)
# 와꾸확인 dim 9 라스트레이어 1 테스트-서브미션 행 180개
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7,random_state=66)
#스케일링
print(test_set)
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = StandardScaler()
# x_train2=pd.DataFrame(scaler.fit_transform(x_train),columns=x_train.columns)
# x_test2=pd.DataFrame(scaler.transform(x_test),columns=x_test.columns)
# test_set2=pd.DataFrame(scaler.transform(test_set),columns=test_set.columns)
# print(test_set2)
# scaler = MinMaxScaler()
# x_train=pd.DataFrame(scaler.fit_transform(x_train2),columns=x_train2.columns)
# x_test=pd.DataFrame(scaler.transform(x_test2),columns=x_test2.columns)
# test_set=pd.DataFrame(scaler.transform(test_set2),columns=test_set.columns)
# print(test_set)
# print(x_test['Unemployment'])

# ef = pd.DataFrame(test_set, columns=['Unemployment'])
# df2 = pd.DataFrame(x_train, columns=['Promotion5'])
# ef2 = pd.DataFrame(test_set, columns=['Promotion5'])
# df3 = pd.DataFrame(x_train, columns=['Promotion4'])
# ef3 = pd.DataFrame(test_set, columns=['Promotion4'])
# df4 = pd.DataFrame(x_train, columns=['Promotion3'])
# ef4 = pd.DataFrame(test_set, columns=['Promotion3'])
# df5 = pd.DataFrame(x_train, columns=['Promotion2'])
# ef5 = pd.DataFrame(test_set, columns=['Promotion2'])
# df6 = pd.DataFrame(x_train, columns=['Promotion1'])
# ef6 = pd.DataFrame(test_set, columns=['Promotion1'])
# df7 = pd.DataFrame(x_train, columns=['Fuel_Price'])
# ef7 = pd.DataFrame(test_set, columns=['Fuel_Price'])
# df8 = pd.DataFrame(x_train, columns=['Temperature'])
# ef8 = pd.DataFrame(test_set, columns=['Temperature'])
# x_train['Unemployment'] = scaler.fit_transform(x_train['Unemployment'])
# x_test['Unemployment'] = scaler.fit_transform(x_train['Unemployment'])

# print(x_test['Unemployment'])


# print(np.min(test_set['Unemployment']))
# print(np.min(x_train['Unemployment']))
# print(np.max(x_train['Unemployment']))
# print(np.max(test_set['Unemployment']))
# 4.077
# 4.077
# 14.313
# 14.313
# 값이 똑같이 나옴 좀 이상하지만 걍 진행;

print(test_set)
print(x)

#함수모델
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Dropout, Dense, Conv2D, MaxPooling2D, Flatten, Input
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime as dt
date = dt.datetime.now()
date = date.strftime('%y%m%d %H%M')
filename = '{epoch:04d} {val_loss:.4f}.hdf5'
es = EarlyStopping(monitor = 'val_loss', mode='min',restore_best_weights=True, patience=10)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', save_best_only=True, filepath = ''.join(['./_ModelCheckPoint/dacon_shop/1.hdf5']))
# 일단 덴스모델로 돌리고 이후  shape로 전환
x_train = x_train.reshape(-1,2,2,1)
x_test = x_test.reshape(-1,2,2,1)
test_set=test_set.reshape(-1,2,2,1
                          )
input1 = Input(shape=(2,2,1))
dense1 = Conv2D(filters=132, kernel_size=(2,2), activation='linear', padding='same')(input1)
drop2 = Dropout(0.2)(dense1)
dense2 = Conv2D(filters=100, kernel_size=(2,2),activation='relu', padding='same')(drop2)
dense3 = Conv2D(filters=10, kernel_size=(2,2),activation='sigmoid', padding='same')(dense2)
dense4 = Conv2D(filters=100, kernel_size=(2,2),activation='relu', padding='same')(dense3)
drop3 = Dropout(0.2)(dense4)
flat = Flatten()(drop3)
dense5 = Dense(units=10, activation='sigmoid')(flat)
output1 = Dense(units=1, activation='linear')(dense5)
model = Model(inputs=input1, outputs=output1)
model.summary()
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_1 (InputLayer)         [(None, 9)]               0
# _________________________________________________________________
# dense (Dense)                (None, 10)                100
# _________________________________________________________________
# dense_1 (Dense)              (None, 10)                110
# _________________________________________________________________
# dense_2 (Dense)              (None, 10)                110
# _________________________________________________________________
# dropout (Dropout)            (None, 10)                0
# _________________________________________________________________
# dense_3 (Dense)              (None, 10)                110
# _________________________________________________________________
# dense_4 (Dense)              (None, 10)                110
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 10)                0
# _________________________________________________________________
# dense_5 (Dense)              (None, 1)                 11
# =================================================================
# Total params: 551
# Trainable params: 551
# Non-trainable params: 0
#모델세이브는 fit 다음에
# import time
# model.compile(loss = 'mse', optimizer='adam', metrics=['accuracy', 'mae'])
# start_time = time.time()
# hist = model.fit(x_train,y_train,epochs=10,batch_size=10,verbose=1,callbacks=[es,mcp],validation_split=0.2)
# end_time = time.time()-start_time
# model.save('./_save/dacon_shop1.h5') 
model = load_model('./_save/dacon_shop1.h5')

# 평가및 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)#답을 도출하기보단 스스로의 셋을 검증하기 위한 값
y_summit = model.predict(test_set)
print(y_predict)
print(y_summit)
#원핫인코딩 트레인셋만 적용해서 테스트셋넣은 y서밋값이 에러남 
#답안지 입력
submission['Weekly_Sales'] = y_summit
submission.to_csv(path +'submission.csv')

from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_predict)

print('loss',loss,'r2',r2)
# import matplotlib.pyplot as plt
# plt.grid()
# plt.figure(figsize=(9.6))
# plt.legend(loc='upper left')
# plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
# plt.plot(hist.history['val_loss'], marker='.', c='red', label='val_loss')
# plt.title('loss')
# plt.xlabel('loss')
# plt.ylabel('epoch')
# plt.show()




