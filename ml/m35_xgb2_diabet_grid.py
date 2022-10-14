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

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=234,)

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
    'n_estimators':[100,200], # 2
              'learning_rate':[0.1,0.3,0.7], # 3
              'max_depth': [2,3,4], # 3
              'gamma' : [0.1,0.9,1], # 4 
              'min_child_weight' : [5], # 1
              'subsample' : [0.1,0.5,0.7,1], # 4
              'colsample_bytree' : [0.3,0.5,0.7,1], # 4
              'colsample_bylevel': [0.3,0.5,0.7,1],# 4
              'colsample_bynode': [0.3,0.5,0.7,1],# 4
              'alpha' : [ 0.001, 1 ,2 ,10], # 3
              'lambda' : [0.001, 1 ,2 ,10] # 3
}

#2.모델
xgb = XGBRegressor(random_state=238)

model = GridSearchCV(xgb,Parameters,cv=kfold,n_jobs=-1)

start = time.time()
model.fit(x_train,y_train)
end = time.time()

print('params',model.best_params_)
print('score',model.best_score_)
print('time',end-start)
from sklearn.metrics import r2_score
print(r2_score(y_test,(model.predict(x_test))))

# params {'alpha': 0.001, 'colsample_bylevel': 0.5, 'colsample_bynode': 0.3, 'colsample_bytree': 0.5, 'gamma': 0.1, 'lambda': 0.001, 'learning_rate': 0.1, 'max_depth': 4, 'min_child_weight': 5, 'n_estimators': 200, 'subsample': 1}
# score 0.9802197802197803
# time 8344.525100708008
# 0.956140350877193