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
from imblearn.over_sampling import SMOTE
smt = SMOTE(random_state=2379,k_neighbors=3)
print(x_train.shape,y_train.shape)
smotestart = time.time()
x_train, y_train = smt.fit_resample(x_train,y_train)
endsmote = time.time()-smotestart
print(x_train.shape,y_train.shape)
print('smotetime',endsmote)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=234)



#2.모델
model = XGBClassifier(random_state=238)


start = time.time()
model.fit(x_train,y_train)
end = time.time()

print('time',end-start)
from sklearn.metrics import accuracy_score, f1_score
print(accuracy_score(y_test,(model.predict(x_test))))
print(f1_score(y_test,(model.predict(x_test))))


# time 0.07396769523620605
# 0.9473684210526315
# 0.9583333333333334

# (455, 30) (455,)
# (570, 30) (570,)
# smotetime 8.469122648239136
# time 0.1575789451599121
# 0.9736842105263158
# 0.979020979020979

