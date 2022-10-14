import tensorflow as tf
tf.random.set_seed(366)
#시퀀셜, 함수모델 모두 적용
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Input
#데이터불러오기
import numpy as np
import pandas as pd
path = './_data/kaggle_titanic/'
filename = '{epoch:04d} {val_loss:.4f}.hdf5'
import datetime as dt
date = dt.datetime.now()
date = date.strftime('%m%d %H%M')
import time as t
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler,StandardScaler,RobustScaler
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
#걍들어가자
test_set = pd.read_csv(path +'test.csv', index_col=0)
train_set = pd.read_csv(path +'train.csv',index_col=0)
submission = pd.read_csv(path +'gender_submission.csv',index_col=0)
#테스트셋과 x 같이 처리, 테스트 스플릿 후 스케일러
#x,y 설정
# print(test_set.shape, submission.shape) #(418, 11) (418, 2) 행수 맞고 칼럼확인

# print(test_set.columns, submission.columns) 
# Index(['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',
#        'Ticket', 'Fare', 'Cabin', 'Embarked'],
#       dtype='object') Index(['PassengerId', 'Survived'], dtype='object')
#패신저 아이디 삭제 index_col=0 subvived 답안지 확인

#트레인셋 칼럼 확인
# print(train_set.columns)
# Index(['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket',
#        'Fare', 'Cabin', 'Embarked'],
#       dtype='object')

#x,y나누기 전에 오브젝트 타입 및 결측치 확인
# print(train_set.describe()) 평균치등 나왔으나 잘모르겠음
# print(train_set.isnull().sum())
# Survived      0
# Pclass        0
# Name          0
# Sex           0
# Age         177
# SibSp         0
# Parch         0
# Ticket        0
# Fare          0
# Cabin       687
# Embarked      2
# dtype: int64   
# 전부 int타입?

# print(train_set.info())
# Data columns (total 11 columns):
#  #   Column    Non-Null Count  Dtype
# ---  ------    --------------  -----
#  0   Survived  891 non-null    int64
#  1   Pclass    891 non-null    int64
#  2   Name      891 non-null    object
#  3   Sex       891 non-null    object
#  4   Age       714 non-null    float64
#  5   SibSp     891 non-null    int64
#  6   Parch     891 non-null    int64
#  7   Ticket    891 non-null    object
#  8   Fare      891 non-null    float64
#  9   Cabin     204 non-null    object
#  10  Embarked  889 non-null    object
# dtypes: float64(2), int64(4), object(5)
#name, sex ticket cabin embarked object 타입
#sex, object 원핫인코더처리
pd.set_option('max_columns', None)
train_set = pd.get_dummies(train_set, columns=['Sex', 'Embarked'])
test_set = pd.get_dummies(test_set, columns=['Sex', 'Embarked'])
#x에 관한 것으로 테스트셋도 같이 처리
# print(train_set)
# print(test_set)
#티켓 네임빼기, x에 관한 것으몰 테스트셋도
# print(x)
# print(y)
# print(x.isnull().sum())
# Pclass          0
# Age           177
# SibSp           0
# Parch           0
# Fare            0
# Cabin         687
# Sex_female      0
# Sex_male        0
# Embarked_C      0
# Embarked_Q      0
# Embarked_S      0
#age, cabin값 빼기로함
train_set = train_set.drop(['Name','Ticket','Age','Cabin'], axis=1)
test_set = test_set.drop(['Name','Ticket','Age','Cabin'], axis=1)
print(train_set.columns)
x= train_set.drop(['Survived'], axis=1)
y=train_set['Survived']
#xy나눴으니 테스트 트레인셋 분리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, random_state=88)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



#수직, 함수모델 구현
#와꾸확인
print(x_train.shape, y_test.shape)

# model= Sequential()
# model.add(Dense(units=5, input_shape=(9,)))
# model.add(Dense(20, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(1, activation='sigmoid'))
# model.summary()
#ㅁ델 웨이트는 핏트 끝나고
#함수
# input1 = Input(shape=(9,))
# dense1 = Dense(5)(input1)
# dense2 = Dense(20, activation='relu')(dense1)
# drop1 = Dropout(0.2)(dense2)
# output1 = Dense(1, activation='sigmoid')(drop1)
# model = Model(inputs=input1, outputs=output1)
# model.summary()
model = load_model('./_save/titanic5/5.h5')
filepath = './_ModelCheckPoint/titanic5/'
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=10,restore_best_weights=True)
mcp = ModelCheckpoint(filepath=''.join([filepath,date,filename]),monitor='val_loss',verbose=1, save_best_only=True, mode='auto')
hist = model.fit(x_train,y_train, epochs=100, verbose=1, callbacks=[es,mcp], validation_split=0.1)
# model.save('./_save/titanic5/5.h5')
# model.save_weights('./_save/titanic5/weights/5.h5')

#평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_summit = model.predict(test_set)
print(np.max(x_train))
print(np.min(x_train))
print(np.max(x_test))
print(np.min(x_test))
print(submission.shape, test_set.shape)
print(y_predict)
print(y_summit)
# y_predict = np.argmax(y_predict, axis=1)
# y_summit = np.argmax(y_summit, axis=1)
# 칼럼별로 나뉜 값이 아니라 바이너리 크로스엔트로피로 1개만 나오므로 아그맥스가 소용없음
y_predict = y_predict.round()
y_summit = y_summit.round()
print(y_predict)
print(y_summit)

from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
def RMSE(a,b):
    return np.sqrt(mean_squared_error(a,b))
rmse = RMSE(y_test, y_predict)
acc = accuracy_score(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

submission['Survived'] = y_summit

print(submission)
submission.to_csv(path + 'submission.csv')