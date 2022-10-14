import numpy as np
import pandas as pd
from inspect import Parameter
from sklearn.datasets import load_boston
from sklearn.model_selection import GridSearchCV,KFold,StratifiedKFold,train_test_split
from xgboost import XGBClassifier,XGBRegressor
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,RobustScaler,QuantileTransformer,PowerTransformer
import matplotlib.pyplot as plt

#1.data
datasets = load_boston()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)

scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
scaler = MaxAbsScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
scaler = RobustScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
scaler = PowerTransformer(method='yeo-johnson')
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
# scaler = PowerTransformer(method='box-cox')
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
scaler = QuantileTransformer()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

model = LinearRegression()
scalist = [QuantileTransformer(),MinMaxScaler(),PowerTransformer(),RobustScaler(),MaxAbsScaler(),StandardScaler()]
for i in scalist:
    x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=234,)
    x_train = i.fit_transform(x_train)
    x_test = i.transform(x_test)
    model.fit(x_train,y_train)
    pred = model.predict(x_test)
    result = r2_score(y_test,pred)
    name = i.__class__.__name__
    print('{0} : {1}'.format(name,round(result,4)))

#2.모델

# model.fit(x_train,y_train)

# pred = model.predict(x_test)

# result = r2_score(y_test,pred)

# print(round(result,4))


# 기존 스케일러 linear
# 0.8086

# 그냥
# 0.7791

# b log 후
# 0.7792

# crime log후
# 0.7784

# zn
# 0.7759

# tax
# 0.7784