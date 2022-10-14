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
    'n_estimators':[100],
    'learning_rate':[0.1],
    'max_depth':[None],
    'gamma': [0],  
    'min_child_weight':[5], 
    'subsample':[1], #디폴트1 0~1
    'colsample_bytree':[0.1], 
    'colsample_bylevel':[1], 
    'colsample_bynode':[1], 
    'reg_alpha':[0], # l1 정규화 절대값으로 exploding 방지 0~inf/ alpha라고해도됨 디폴트0
    # 'reg_lambda':[0,0.1,0.001,0.00000001,100,10000,100000] # l2 정규화 제곱으로 exploding 방지 0~inf 디폴트1 lambda
}

#2.모델
model = XGBClassifier(random_state=238,n_estimators=1000,
                      learning_rate=0.1,
                      max_depth=3,
                      gemma=1)

# model = GridSearchCV(xgb,Parameters,cv=kfold,n_jobs=-1)

start = time.time()
model.fit(x_train,y_train, early_stopping_rounds=10,eval_set=[(x_test,y_test)],
          eval_metric='error',#다중분류나 회귀모델에 따라 다르다rmse error merror\
        #   eval_set = )
)
end = time.time()

# print('params',model.best_params_)
# print('score',model.best_score_)
print('time',end-start)
print('score',model.score(x_test,y_test))
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,(model.predict(x_test))))

import matplotlib.pyplot as plt
hist = model.evals_result()
print(hist)
print(type(hist))
print(hist.items())
# print(hist[0])
# print(hist[1])
print(hist['validation_0'])
print(hist.values)
d = hist['validation_0'].items()
print(d)
x, y = zip(*d)
print(x)
print(y)
# plt.plot(x, y)
# plt.show()
x = np.array(x)
for i in range(len(y)):
  x = x.append(x)
  plt.plot(x,y[i])
plt.show()


# def plot_feature_importances_dataset(model):
#     n_features = hist.shape[1]
#     plt.barh(np.arange(n_features),hist,align='center')
#     # plt.yticks(np.arange(n_features),datasets.feature_names)
#     plt.xlabel('feature importances')
#     plt.ylabel('features')
#     plt.ylim(-1,n_features)

# plt.subplot(4,1,1)
# plot_feature_importances_dataset(hist)