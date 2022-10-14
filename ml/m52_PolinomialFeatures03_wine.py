import numpy as np
import pandas as pd
from sklearn.datasets import load_boston, load_breast_cancer,fetch_covtype,load_wine,load_iris,load_digits
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
#1. 데이터
datasets = fetch_covtype()
datasets = load_breast_cancer()
datasets = load_digits()
datasets = load_wine()
# datasets = load_iris()
import xgboost as xg
print(xg.__version__)
x = datasets.data
y = datasets.target
print(x.shape) #(581012, 54)
from sklearn.preprocessing import MinMaxScaler,StandardScaler,PolynomialFeatures
pf = PolynomialFeatures(degree=2,include_bias=False)
x = pf.fit_transform(x)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# pca = PCA(n_components=10)
# x = pca.fit_transform(x)
lda = LinearDiscriminantAnalysis(n_components = 2)
lda.fit(x,y)
x = lda.transform(x)
# pca_EVR = pca.explained_variance_ratio_
# cumsum = np.cumsum(pca_EVR)
# y = pd.get_dummies(y)
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8,random_state=234,stratify=y)
print(np.unique(y_train,return_counts=True)) #(array([1, 2, 3, 4, 5, 6, 7]), array([169472, 226640,  28603,   2198,   7594,  13894,  16408],

#2.모델
from xgboost import XGBClassifier,XGBRegressor
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from lightgbm import LGBMClassifier,LGBMRegressor
from catboost import CatBoostClassifier,CatBoostRegressor
from sklearn.metrics import r2_score,accuracy_score
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
lr = LogisticRegression()
knn = KNeighborsClassifier(n_neighbors=8)
xg = XGBClassifier()
lg = LGBMClassifier()
cat = CatBoostClassifier(verbose=0)


model = VotingClassifier(estimators=[
    ('LR',lr),
    ('KNN',knn),
    ('XG',xg),
    ('LG',lg),
    ('CAT',cat)],
                         voting='hard')
classifier = [lr,knn,xg,lg,cat]
for model2 in classifier:
    model2.fit(x_train,y_train)
    pred = model2.predict(x_test)
    score2 = accuracy_score(y_test,pred)
    class_name = model2.__class__.__name__
    print('{0} 정확도: {1:.4f}'.format(class_name,score2))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x_train, y_train, cv=5, scoring ='accuracy')
print(np.mean(scores))
#3.훈련
# import time
# start = time.time()
# model.fit(x_train, y_train)
# print('시간',time.time()-start)
# #4.평가 예측
# result = model.score(x_test, y_test)
# print('결과',result)
# print(np.unique(y))




# no pca lda
# 시간 0.5370004177093506
# 결과 0.9722222222222222

# pca 10
# x.shape(178, 13)
# 시간 0.5436191558837891
# 결과 0.9166666666666666

# lda 2
# y np.unique [0 1 2]
# 시간 0.5383071899414062
# 결과 1.0

# xgb
# 시간 1.0660011768341064
# 결과 1.0

# xgb bagging
# 시간 8.06783151626587
# 결과 1.0

# voting
# LogisticRegression 정확도: 1.0000
# KNeighborsClassifier 정확도: 1.0000
# XGBClassifier 정확도: 1.0000
# LGBMClassifier 정확도: 1.0000
# CatBoostClassifier 정확도: 1.0000