parameters = [
     {'n_estimators':[100,200],'max_depth':[1,3,2,5,10]},
     {'max_depth':[6,8,10,12],'min_samples_leaf':[1234,23,41,22]},
     {'min_samples_leaf':[3,5,7,10],'min_samples_split':[1,23,24,2,3,5]},
     {'min_samples_split':[2,3,5,10],'n_estimators':[400,20]},
     {'n_jobs':[-1,2,4],'n_estimators':[159,1278,2345,1234],'min_samples_leaf':[6,1,80],'min_samples_split':[1795,13947,149875,19387]}
]
# Fitting 5 folds for each of 202 candidates, totalling 1010 fits

#2개 이상 엮을 것
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
from sklearn.svm import LinearSVC, SVC #리니어 서포트 벡터 
from sklearn.linear_model import Perceptron,LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score,GridSearchCV, StratifiedKFold
n_splits = 1000
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=328947)
import time
start = time.time()
# model = SVC(C=1,kernel='linear',degree =3)
model = GridSearchCV(RandomForestClassifier(), parameters,cv=kfold,verbose=1,refit=True,n_jobs=-1)
# model = LinearSVC()
# model = SVC()
# model = Perceptron()
# model = LogisticRegression()
# model = KNeighborsClassifier()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()
model.fit(x_train,y_train)
end = time.time()
print('최적 매개변수',model.best_estimator_)
print('최적 파라',model.best_params_)
print('bestcore',model.best_score_)
result = model.score(x_test,y_test)
print('model.score',result)
from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print('acc',acc)
y_best_best = model.best_estimator_.predict(x_test)
print(y_best_best)
print(round(end-start,4),'cho')
""" 



print(acc,result) """