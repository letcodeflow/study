#r2 0.62 이상
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split,cross_val_score,KFold,GridSearchCV,RandomizedSearchCV
import time
from sklearn.utils import all_estimators
import warnings
from sklearn.metrics import r2_score
warnings.filterwarnings('ignore')
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x)
print(y)
#y는 어차피 결과값으로 나오는 것이기 때문에 여기서 전처리는 필요없다

# print(x.shape)
# print(y.shape)

# print(datasets.feature_names)
# print(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)
n_splits = 5
allAlgorithms = all_estimators(type_filter='regressor')
kfold = KFold(n_splits=n_splits,random_state=234,shuffle=True)
parameters = [
     {'n_estimators':[100,200],'max_depth':[1,3,2,5,10]},
     {'max_depth':[6,8,10,12],'min_samples_leaf':[1234,23,41,22]},
     {'min_samples_leaf':[3,5,7,10],'min_samples_split':[1,23,24,2,3,5]},
     {'min_samples_split':[2,3,5,10],'n_estimators':[400,20]},
     {'n_jobs':[-1,2,4],'n_estimators':[159,1278,2345,1234],'min_samples_leaf':[6,1,80],'min_samples_split':[1795,13947,149875,19387]}
]

model = RandomizedSearchCV(RandomForestClassifier(),parameters,n_jobs=-1,refit=True,cv=kfold)
start = time.time()
model.fit(x_train,y_train)
print(model.best_estimator_)
print(model.best_index_)
print(model.best_params_)
print(model.best_score_)
print(model.best_estimator_.predict(x_test))
print(time.time()-start/60,'bun')
