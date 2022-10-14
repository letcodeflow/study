from sklearn import metrics
from sklearn.datasets import load_boston

data_sets = load_boston()

x = data_sets.data
y = data_sets.target

from sklearn.model_selection import train_test_split,cross_val_score,KFold,StratifiedKFold,GridSearchCV,RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=False)
from sklearn.utils import all_estimators
allAlgorithms = all_estimators(type_filter='regressor')
n_splits=5
kfold = KFold(n_splits=n_splits,shuffle=True, random_state=2398)
from sklearn.experimental import enable_halving_search_cv
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score,GridSearchCV, StratifiedKFold,RandomizedSearchCV,HalvingGridSearchCV,HalvingRandomSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline, Pipeline
pipe = Pipeline([('minmax',MinMaxScaler()),('RF',RandomForestRegressor())],verbose=2)
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
print(model.score(x_test))