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
from xgboost import XGBRegressor
model = XGBRegressor()

# model = GridSearchCV(pipe, parameters,cv=kfold,verbose=1)
import time
start = time.time()
model.fit(x_train,y_train)
print(time.time()-start)
print(model.score(x_test,y_test))
from sklearn.feature_selection import SelectFromModel

thresholds = model.feature_importances_
print(f'====================')
for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh,prefit=True)
    
    selec_x_train = selection.transform(x_train)
    selec_x_test = selection.transform(x_test)
    print(selec_x_train.shape, selec_x_test.shape)

    model2 = XGBRegressor(n_jobs=-1,random_state=238,n_estimators=1000,
                      learning_rate=0.1,
                      max_depth=3,
                      gemma=1)
    
    model2.fit(selec_x_train,y_train)
    y_pred = model2.predict(selec_x_test)
    score = r2_score(y_test,y_pred)
    print("thresh=%.3f, n=%d, r2: %.2f%%"
          % (thresh, selec_x_train.shape[1],score*100))
