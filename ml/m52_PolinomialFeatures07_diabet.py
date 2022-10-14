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
from sklearn.preprocessing import MinMaxScaler,StandardScaler,PolynomialFeatures
pf = PolynomialFeatures(degree=2,include_bias=False)
x = pf.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=234)

scaler = MinMaxScaler()
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

kfold = KFold(n_splits=5, shuffle=True, random_state=234)
#2.XGBregressor
from xgboost import XGBClassifier,XGBRegressor
from sklearn.ensemble import VotingClassifier,VotingRegressor
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from lightgbm import LGBMClassifier,LGBMRegressor
from catboost import CatBoostClassifier,CatBoostRegressor
from sklearn.metrics import r2_score,accuracy_score
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
lr = LinearRegression()
knn = KNeighborsRegressor(n_neighbors=8)
xg = XGBRegressor()
lg = LGBMRegressor()
cat = CatBoostRegressor(verbose=0)


model = VotingRegressor(estimators=[
    ('LR',lr),
    ('KNN',knn),
    ('XG',xg),
    ('LG',lg),
    ('CAT',cat)],
                         )
classifier = [lr,knn,xg,lg,cat]
for model2 in classifier:
    model2.fit(x_train,y_train)
    pred = model2.predict(x_test)
    score2 = r2_score(y_test,pred)
    class_name = model2.__class__.__name__
    print('{0} 정확도: {1:.4f}'.format(class_name,score2))
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x_train, y_train, cv=5, scoring ='r2')
print(np.mean(scores))
# print('r2',r2_score(y_test,model.predict(x_test)))

# xgb = XGBRegressor(random_state=238)

# from sklearn.ensemble import BaggingClassifier,BaggingRegressor
# # model = BaggingRegressor(XGBRegressor(), n_estimators=100, random_state=23)

# start = time.time()
# model.fit(x_train,y_train)
# end = time.time()

# # print('params',model.best_params_)
# # print('score',model.best_score_)
# print('time',end-start)
# from sklearn.metrics import r2_score
# print(r2_score(y_test,(model.predict(x_test))))

# xgb grid
# score 0.2443777933346439
# time 3.1116368770599365
# 0.31370474417988337

# xgb bagging
# time 10.85739016532898
# 0.42723166481153063

# voting
# LinearRegression 정확도: 0.5119
# KNeighborsRegressor 정확도: 0.5014
# XGBRegressor 정확도: 0.2690
# LGBMRegressor 정확도: 0.3066
# CatBoostRegressor 정확도: 0.3867

# polinomaial 
# LinearRegression 정확도: 0.5046
# KNeighborsRegressor 정확도: 0.4908
# XGBRegressor 정확도: 0.3372
# LGBMRegressor 정확도: 0.3105
# CatBoostRegressor 정확도: 0.4095
# 0.4751770730755556
