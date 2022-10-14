from inspect import Parameter
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV,KFold,StratifiedKFold,train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from xgboost import XGBClassifier,XGBRegressor
import time
import numpy as np

#1.data
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=234,stratify=y)

scaler = MinMaxScaler()
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=234)

# Parameters = {
#     'n_estimators':[100,200,300,400,500,1000] #디폴트100 1~inf/정수
# }
# Parameters = {
#     'n_estimators':[100],
#     'learning_rate':[0.1,0.2,0.3,0.4,0.5,1,0.01] #디폴트 0.3 /0~1, 다른이름 eta
# }
# Parameters = {
#     'n_estimators':[100],
#     'learning_rate':[0.1],
#     'max_depth':[None,2,3,4,5,6,7,8,9,10], #None 무한대 default 6 / 0~inf/ 정수
# }
# Parameters = {
#     'n_estimators':[100],
#     'learning_rate':[0.1],
#     'max_depth':[None],
#     'gamma': [0], #loss값 조각 
#     'min_child_weight':[5], #디폴트 1 /0~inf
# }
Parameters = {
    'n_estimators':[100,200], # 2
              'learning_rate':[0.1,0.3,0.7], # 3
              'max_depth': [2,3,4], # 3
              'gamma' : [0.1,0.9,1], # 4 
              'min_child_weight' : [5], # 1
              'subsample' : [0.1,0.5,0.7,1], # 4
              'colsample_bytree' : [0.3,0.5,0.7,1], # 4
              'colsample_bylevel': [0.3,0.5,0.7,1],# 4
              'colsample_bynode': [0.3,0.5,0.7,1],# 4
              'alpha' : [ 0.001, 1 ,2 ,10], # 3
              'lambda' : [0.001, 1 ,2 ,10] # 3
}

#2.모델
model = XGBClassifier(random_state=238)

# model = GridSearchCV(xgb,Parameters,cv=kfold,n_jobs=-1)

start = time.time()
model.fit(x_train,y_train)
end = time.time()

print('time',end-start)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,(model.predict(x_test))))

from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
thresholds = model.feature_importances_
print(f'====================')
for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh,prefit=True)
    
    selec_x_train = selection.transform(x_train)
    selec_x_test = selection.transform(x_test)
    print(selec_x_train.shape, selec_x_test.shape)

    model2 = XGBClassifier(n_jobs=-1,random_state=238,n_estimators=1000,
                      learning_rate=0.1,
                      max_depth=3,
                      gemma=1,tree_method='gpu_hist',predictor='gpu_predictor',gpu_id=0)
    
    model2.fit(selec_x_train,y_train)
    y_pred = model2.predict(selec_x_test)
    score = accuracy_score(y_test,y_pred)
    print("thresh=%.3f, n=%d, r2: %.2f%%"
          % (thresh, selec_x_train.shape[1],score*100))