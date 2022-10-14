import tensorflow as tf
tf.random.set_seed(137)

import numpy as np
from sklearn.datasets import load_iris
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split,HalvingGridSearchCV

#1.데이터
datasets = load_iris()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, shuffle=True, random_state=137)
from sklearn.svm import LinearSVC, SVC #리니어 서포트 벡터 
from sklearn.linear_model import Perceptron,LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score,GridSearchCV, StratifiedKFold
import sklearn
print(sklearn.__version__)
n_splits = 5
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=328947)
parameters = [
    {"C":[1,10,100,1000], "kernel":["linear"],"degree":[3,4,5]},
    {"C":[1,10,100], "kernel":["rbf"],"gamma":[0.001,0.00001]},
    {"C":[1,10,100,1000], "kernel":["sigmoid"],"gamma":[0.01,0.001,0.00001],
    "degree":[3,4]},
]
import time
start = time.time()
model = SVC(C=1,kernel='linear',degree =3)
model = HalvingGridSearchCV(SVC(), parameters,cv=kfold,verbose=1,refit=True,n_jobs=-1)
model.fit(x_train,y_train)
end = time.time() 
print('최적 매개변수',model.best_estimator_)
print('최적 파라',model.best_params_)
print('bestcore',model.best_score_)
y_best_best = model.best_estimator_.predict(x_test)
print(y_best_best)
print(round(end-start,4),'cho')
""" 



print(acc,result) """