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
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import KFold, cross_val_score,GridSearchCV, StratifiedKFold,RandomizedSearchCV,HalvingGridSearchCV,HalvingRandomSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline, Pipeline
pipe = Pipeline([('minmax',MinMaxScaler()),('RF',RandomForestClassifier())],verbose=2)
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,HalvingGridSearchCV,HalvingRandomSearchCV,KFold,StratifiedKFold
parameters = [
     {'RF__n_estimators':[100,200],'RF__max_depth':[1,3,5,10]},
     {'RF__max_depth':[6,8,12],'RF__min_samples_leaf':[23,41,22]},
     {'RF__min_samples_leaf':[3,5,7,10],'RF__min_samples_split':[1,2,3,5]},
     {'RF__min_samples_split':[2,3,5,10],'RF__n_estimators':[400,20]},
     {'RF__n_jobs':[-1,2,4],'RF__n_estimators':[159,12,23],'RF__min_samples_leaf':[6,1,80],'RF__min_samples_split':[17,13,19]}
]
n_splits = 10
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=328947)

model = GridSearchCV(pipe, parameters,cv=kfold,verbose=1)
import time
start = time.time()
model.fit(x_train,y_train)
print(time.time()-start)
# predict = model.best_estimator_.predict(x_test)
# print(predict)
# print(model.best_estimator_)
# print(model.best_index_)
# print(model.best_params_)
# print(model.best_score_)