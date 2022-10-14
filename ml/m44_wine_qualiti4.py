#csv로 만들기
# 3 d dsutdy data _data 에 복사

# sklearn은 데이터셋의 일부였으므로 지금 파일은 원래의 다중분류/ 퀄리티 = y


import numpy as np
import pandas as pd

# 카톡->원드라이브 업로드=깃허브 페치-재로딩
#read
path = '서밋루트'
dataset = pd.read_csv('https://github.com/letcodesing/study/raw/main/_data/winequality-white.csv',index_col=None,
                      header=0,
                      sep=';')
# pd.set_option('display.max_rows',None)                      
print(dataset.describe())
# dataset = dataset.to_numpy()
print(type(dataset))
#상관계수 뽑아보기
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(dataset.corr(),annot=True)
# plt.show()
# 상관계수 - 전부 삭제
# print(dataset.columns)
# Index(['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
#        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
#        'pH', 'sulphates', 'alcohol', 'quality'],
#       dtype='object')

dataset.drop(columns=['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','total sulfur dioxide', 'density'],axis=1,inplace=True)
# y = qutality고유값 추출
# print(dataset['quality'].value_counts())
# 6.0    2208
# 5.0    1460
# 7.0     883
# 8.0     175
# 4.0     164
# 3.0      20
# 9.0       5
# Name: quality, dtype: int64
# 고유값 7개 데이터타입 인트
#스케일러 적용전 단위 보기위해 프린트
# print(dataset)
# free sulfur dioxide, 십자리수
# sulphates 소수점이하
# 해당컬럼 따로 처리
#그전에 난값 처리
print(dataset.isnull().sum())
# free sulfur dioxide    0
# pH                     1
# sulphates              2
# alcohol                2
# quality                2
# 난값 most frequent 처리
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy = 'most_frequent')
dataset = imp.fit_transform(dataset)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# dataset[:,-1] = le.fit_transform(dataset[:,-1])
# print(np.unique(dataset[:,-1]))

# print(type(dataset))
# 넘파이
# print(dataset[:5,:1])
# 첫번째열에 free sulfur dioxide 확인
dataset[:,:1] = scaler.fit_transform(dataset[:,:1])
dataset[:,1:2] = scaler.fit_transform(dataset[:,1:2])
dataset[:,2:3] = scaler.fit_transform(dataset[:,2:3])
dataset[:,3:4] = scaler.fit_transform(dataset[:,3:4])
#y값 빼고 스케일링
# print(dataset)
# 모델은 랜포?
#추후 평가 및 과적합 방지, 여러가지 모델 적용르 위해 스플릿
x = dataset[:,:-1]
#프로젝트 모델에서 칼럼증폭 가져옴
# import torch
# from torch import nn
# import torch.nn.functional as F
# print(dataset.shape)
# x = dataset[:,:-1].copy()
# x = x.reshape(-1,4,1)
# x = torch.tensor(x)
# x = F.interpolate(x, scale_factor=150, mode='linear')
# x = x.numpy()
# print(x.shape)
# x = x.reshape(-1,600)
# acc값 같음 소용없음
new=[]
for i in dataset[:,-1]:
    if i <= 5:
        new += [0]
    elif i ==6:
        new += [1]
    else:
        new += [2]
print(np.unique(new,return_counts=True))
from sklearn.model_selection import train_test_split,RandomizedSearchCV,GridSearchCV
x_train, x_test, y_train, y_test = train_test_split(x,new,random_state=23786,train_size=0.8, stratify=dataset[:,-1])
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
# xgb = XGBClassifier()
# xgb.fit(x_train,y_train)
# pred = xgb.predict(x_test)
rf = RandomForestClassifier()
rf.fit(x_train,y_train)
pred = rf.predict(x_test)

#xgb 쓸때만 라벨인코딩 y값이 0~부터 차례로 나오는 것이 아니라 각각의 고유값을 가지기 때문
from sklearn.metrics import accuracy_score,f1_score
print('acc',accuracy_score(y_test,pred),f1_score(y_test,pred,average='macro'))
print('acc',accuracy_score(y_test,pred),f1_score(y_test,pred,average='micro'))


def outlier(data):
    q1,q2,q3 = np.percentile(data,[25,50,75])
    print(q1,q2,q3)
    iqr = q3-q1
    print(iqr)
    lower_bound = q1-(iqr*1.5)
    upper_bound = q3+(iqr*1.5)
    return np.where((data>upper_bound)|(data<lower_bound))

y_out_loc = outlier(dataset[:,-1])
print(len(y_out_loc))