import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8,random_state=234,stratify=y)

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

#3.훈련
import time
start = time.time()
model.fit(x_train, y_train)
print('시간',time.time()-start)
#4.평가 예측

print(x_test.shape, y_test.shape)
# y_test = y_test.reshape(-1,1)
# result = model.score(x_test, y_test)
# print('결과',result)


classifier = [lr,knn,xg,lg,cat]
for model2 in classifier:
    model2.fit(x_train,y_train)
    pred = model2.predict(x_test)
    score2 = accuracy_score(y_test,pred)
    class_name = model2.__class__.__name__
    print('{0} 정확도: {1:.4f}'.format(class_name,score2))

# xxgboost
# 시간 7.320397138595581
# 결과 0.8953985697443267

# no pca
# 시간 5.952320337295532
# 결과 0.8683166527542319

# pca 18
# 시간 4.731928825378418
# 결과 0.8846587437501614

# lda 
# 시간 3.589005708694458
# 결과 0.7874581551250829

# lda compo 2
# 시간 0.7507467269897461
# 결과 1.0

# pca 2
# 시간 0.4982614517211914
# 결과 0.8666666666666667

# no pca lda
# 시간 0.5761358737945557
# 결과 0.9333333333333333

# bagging XGBClassifier
# 시간 9.621065139770508
# 결과 0.9333333333333333

# voting
# LogisticRegression 정확도: 0.9333
# KNeighborsClassifier 정확도: 0.8667
# XGBClassifier 정확도: 0.9333
# LGBMClassifier 정확도: 0.9333
# CatBoostClassifier 정확도: 0.9333