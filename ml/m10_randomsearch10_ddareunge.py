#1. 데이터
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score,KFold,cross_val_predict,GridSearchCV,RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')
pd.__version__
path = 'c:/study/_data/ddareunge/'
train_set = pd.read_csv(path + 'train.csv', index_col=0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)


train_set = train_set.fillna(0) # 결측치 0으로 채움

x = train_set.drop(['count'], axis=1)
y = train_set['count']


x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.8)

allAlgoithms = all_estimators(type_filter='regressor')

from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = KFold(n_splits=n_splits,random_state=2371,shuffle=True)
parameters = [
     {'n_estimators':[100,200],'max_depth':[1,3,2,5,10]},
     {'max_depth':[6,8,10,12],'min_samples_leaf':[1234,23,41,22]},
     {'min_samples_leaf':[3,5,7,10],'min_samples_split':[1,23,24,2,3,5]},
     {'min_samples_split':[2,3,5,10],'n_estimators':[400,20]},
     {'n_jobs':[-1,2,4],'n_estimators':[159,1278,2345,1234],'min_samples_leaf':[6,1,80],'min_samples_split':[1795,13947,149875,19387]}
]
import time
from sklearn.ensemble import RandomForestRegressor
model = RandomizedSearchCV(RandomForestRegressor(),parameters,n_jobs=-1,refit=True,cv=kfold)
start = time.time()
model.fit(x_train,y_train)
print(model.best_estimator_)
print(model.best_index_)
print(model.best_params_)
print(model.best_score_)
print(model.best_estimator_.predict(x_test))
print(time.time()-start,'cho')
