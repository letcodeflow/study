from inspect import Parameter
from sklearn.datasets import load_diabetes
from sklearn.model_selection import GridSearchCV,KFold,StratifiedKFold,train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from xgboost import XGBClassifier,XGBRegressor
import time
import numpy as np

#1.data
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=234)

scaler = MinMaxScaler()
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

kfold = KFold(n_splits=5, shuffle=True, random_state=234)

# Parameters = {
#     'n_estimators':[100,200,300,400,500,1000] #디폴트100 1~inf/정수
# }
# Parameters = {
#     'n_estimators':[100],
#     'learning_rate':[0.1,0.2,0.3,0.4,0.5,1,0.01] #디폴트 0.3 /0~1, 다른이름 eta
# }
# Parameters = {
#     'n_estimators':[100],
#     'learning_rate':[0.1],
#     'max_depth':[None,2,3,4,5,6,7,8,9,10], #None 무한대 default 6 / 0~inf/ 정수
# }
# Parameters = {
#     'n_estimators':[100],
#     'learning_rate':[0.1],
#     'max_depth':[None],
#     'gamma': [0], #loss값 조각 
#     'min_child_weight':[5], #디폴트 1 /0~inf
# }
Parameters = {
    'n_estimators':[100],
    'learning_rate':[0.1],
    'max_depth':[None],
    'gamma': [0],  
    'min_child_weight':[5], 
    'subsample':[1], #디폴트1 0~1
    'colsample_bytree':[0.1], 
    'colsample_bylevel':[1], 
    'colsample_bynode':[1], 
    'reg_alpha':[0], # l1 정규화 절대값으로 exploding 방지 0~inf/ alpha라고해도됨 디폴트0
    # 'reg_lambda':[0,0.1,0.001,0.00000001,100,10000,100000] # l2 정규화 제곱으로 exploding 방지 0~inf 디폴트1 lambda
}

#2.XGBregressor
xgb = XGBRegressor(random_state=238)

model = GridSearchCV(xgb,Parameters,cv=kfold,n_jobs=-1)
from sklearn.ensemble import BaggingClassifier,BaggingRegressor
model = BaggingRegressor(XGBRegressor(), n_estimators=100, random_state=23)

start = time.time()
model.fit(x_train,y_train)
end = time.time()

# print('params',model.best_params_)
# print('score',model.best_score_)
print('time',end-start)
from sklearn.metrics import r2_score
print(r2_score(y_test,(model.predict(x_test))))

# xgb grid
# score 0.2443777933346439
# time 3.1116368770599365
# 0.31370474417988337

# xgb bagging
# time 10.85739016532898
# 0.42723166481153063


