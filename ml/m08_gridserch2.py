#머신 러닝 모델 수백개
#통상 100번정도 계산한다
#컴파일이 핏이 통합돼있다
#evalu 대신 score
#r2,acc가 자동으로 선택된다
import tensorflow as tf
tf.random.set_seed(137)
#텐서플로로 웨이트값에 처음 난수를 이렇게 주겠다 데이터값에 주는 난수표와는 다름 

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
#1.데이터
datasets = load_iris()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, shuffle=True, random_state=137)
#shuffle=False 일경우 순차데이터로 정렬돼있기때문에 한쪽 특성값이 잘려나가게 된다

# from tensorflow.python.keras import to_categorical
# y_train = to_categorical(y_train, 3)
# y_test = to_categorical(y_test, 3)

# from sklearn.preprocessing import OneHotEncoder
# oh=OneHotEncoder
# y_train = y_train.reshape(-1,1)
# print(y_train)
# y_train = oh.fit(y_train, None)
# print(y_train)
# import pandas as pd
# y_train = pd.get_dummies
from sklearn.svm import LinearSVC, SVC #리니어 서포트 벡터 
from sklearn.linear_model import Perceptron,LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score,GridSearchCV, StratifiedKFold
n_splits = 5
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=328947)
parameters = [
    {"C":[1,10,100,1000], "kernel":["linear"],"degree":[3,4,5]},
    {"C":[1,10,100], "kernel":["rbf"],"gamma":[0.001,0.00001]},
    {"C":[1,10,100,1000], "kernel":["sigmoid"],"gamma":[0.01,0.001,0.00001],
    "degree":[3,4]},
]
model = SVC(C=1,kernel='linear',degree =3)
# model = GridSearchCV(SVC(), parameters,cv=kfold,verbose=1,refit=True,n_jobs=-1)
# model = LinearSVC()
# model = SVC()
# model = Perceptron()
# model = LogisticRegression()
# model = KNeighborsClassifier()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()

model.fit(x_train,y_train)
# print('최적 매개변수',model.best_estimator_)
# print('최적 파라',model.best_params_)
# print('bestcore',model.best_score_)
result = model.score(x_test,y_test)
print('model.score',result)
from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print('acc',acc)
# y_best_best = model.best_estimator_.predict(x_test)
# print(y_best_best)
""" 



print(acc,result) """