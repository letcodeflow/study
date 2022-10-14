#1. 데이터
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score,KFold,cross_val_predict,GridSearchCV,RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')
pd.__version__
path = 'C:/Users/aia200/OneDrive - 한국방송통신대학교/beat/study/_data/ddareunge/'
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
from xgboost import XGBClassifier,XGBRegressor
from sklearn.ensemble import VotingClassifier,VotingRegressor
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from lightgbm import LGBMClassifier,LGBMRegressor
from catboost import CatBoostClassifier,CatBoostRegressor
from sklearn.metrics import r2_score,accuracy_score
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
lr = LinearRegression()
knn = KNeighborsRegressor(n_neighbors=8)
xg = XGBRegressor()
lg = LGBMRegressor()
cat = CatBoostRegressor(verbose=0)


model = VotingRegressor(estimators=[
    ('LR',lr),
    ('KNN',knn),
    ('XG',xg),
    ('LG',lg),
    ('CAT',cat)],
                         )
classifier = [lr,knn,xg,lg,cat]
for model2 in classifier:
    model2.fit(x_train,y_train)
    pred = model2.predict(x_test)
    score2 = r2_score(y_test,pred)
    class_name = model2.__class__.__name__
    print('{0} 정확도: {1:.4f}'.format(class_name,score2))


# model = GridSearchCV(pipe, parameters,cv=kfold,verbose=1)
# import time
# start = time.time()
# model.fit(x_train,y_train)
# print(time.time()-start)
# print(model.score(x_test,y_test))

# xgb
# 0.2199995517730713
# 0.8278279081370832

# xgb bagging
# 15.233698606491089
# 0.8005528500648607

# voting
# LinearRegression 정확도: 0.6018
# KNeighborsRegressor 정확도: 0.6648
# XGBRegressor 정확도: 0.7900
# LGBMRegressor 정확도: 0.7752
# CatBoostRegressor 정확도: 0.7924