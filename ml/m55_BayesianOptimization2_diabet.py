from sklearn.datasets import load_diabetes
from sklearn.model_selection import GridSearchCV,KFold,StratifiedKFold,train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from xgboost import XGBClassifier,XGBRegressor
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import r2_score
#1.data
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=234)

bayesian_params = {
    'max_depth':(6,16),
    'num_leaves':(24,64),
    'min_child_samples':(10,200),
    'min_child_weight':(1,50),
    'subsample':(0.5,1),
    'colsample_bytree':(0.5,1),
    'max_bin':(10,500),
    'reg_lambda':(0.001,10),
    'reg_alpha':(0.01,50)
}
def lgb_hamsu(max_depth, num_leaves, min_child_samples, min_child_weight,subsample,colsample_bytree,max_bin,reg_lambda, reg_alpha):
    params = {
        'n_estimator':500, 'learning_rate':0.02,
        'max_depth': int(round(max_depth)), #위에서 받아온 값의 무조건 정수형
        'num_leaves':int(round(num_leaves)),
        'min_child_samples':int(round(min_child_samples)),
        'min_child_weight':int(round(min_child_weight)),
        'subsample':max(min(subsample,1),0), #어떤 값이든 0~1사이
        'colsample_bytree':max(min(colsample_bytree,1),0),
        'max_bin':max(int(round(max_bin)),10), #10이상만
        'reg_lambda':max(reg_lambda,0), #양수만
        'reg_alpha':max(reg_alpha,0)
    }
    model = LGBMRegressor(**params) #**딕셔너리 형태로 받겠다 * 여러개의 인자를 받겠다
    lg = LGBMRegressor()

    lg.fit(x_train,y_train, eval_set = [(x_train,y_train),(x_test,y_test)],
        eval_metric = 'rmse',
        verbose=0,
        early_stopping_rounds=50)
    pred = lg.predict(x_test)
    result = r2_score(y_test,pred)
    return result
lgb_bo = BayesianOptimization(f=lgb_hamsu,
                              pbounds=bayesian_params,
                              random_state=29837)
lgb_bo.maximize(init_points=5,n_iter=50)
print(lgb_bo.max)


# xgb grid
# score 0.2443777933346439
# time 3.1116368770599365
# 0.31370474417988337

# xgb bagging
# time 10.85739016532898
# 0.42723166481153063

# voting
# LinearRegression 정확도: 0.5119
# KNeighborsRegressor 정확도: 0.5014
# XGBRegressor 정확도: 0.2690
# LGBMRegressor 정확도: 0.3066
# CatBoostRegressor 정확도: 0.3867

# polinomaial 
# LinearRegression 정확도: 0.5046
# KNeighborsRegressor 정확도: 0.4908
# XGBRegressor 정확도: 0.3372
# LGBMRegressor 정확도: 0.3105
# CatBoostRegressor 정확도: 0.4095
# 0.4751770730755556

# batchopti
# 'target': 0.42414247675074834,