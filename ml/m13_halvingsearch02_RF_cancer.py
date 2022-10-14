from tkinter import Grid
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, train_test_split, KFold,GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler #대문자 클래스 약어도 from sklearn.svm import LinearSVC, SVC #리니어 서포트 벡터 
from sklearn.linear_model import Perceptron,LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')
allAlgorithms = all_estimators(type_filter='classifier')
n_splits = 5
kfold = KFold(n_splits=n_splits, random_state=214,shuffle=True)
#1.데이터
data_sets = load_breast_cancer()

x = data_sets['data']
y = data_sets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import KFold, cross_val_score,GridSearchCV, StratifiedKFold,RandomizedSearchCV,HalvingGridSearchCV,HalvingRandomSearchCV
parameters = [
     {'n_estimators':[100,200],'max_depth':[1,3,2,5,10]},
     {'max_depth':[6,8,10,12],'min_samples_leaf':[1234,23,41,22]},
     {'min_samples_leaf':[3,5,7,10],'min_samples_split':[1,23,24,2,3,5]},
     {'min_samples_split':[2,3,5,10],'n_estimators':[400,20]},
     {'n_jobs':[-1,2,4],'n_estimators':[159,1278,2345,1234],'min_samples_leaf':[6,1,80],'min_samples_split':[1795,13947,149875,19387]}
]
model =  HalvingGridSearchCV(RandomForestClassifier(),parameters,cv=kfold,refit=True,n_jobs=-1,verbose=1)
model =  HalvingRandomSearchCV(RandomForestClassifier(),parameters,cv=kfold,refit=True,n_jobs=-1,verbose=1)
import time
start = time.time()
model.fit(x_train,y_train)
print(time.time()-start)
print(predict)
print(model.best_estimator_)
print(model.best_index_)
print(model.best_params_)
print(model.best_score_)
predict = model.best_estimator_.predict(x_test)