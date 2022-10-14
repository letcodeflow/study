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

#1.data
datasets = load_boston()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=234,)

model = make_pipeline(PolynomialFeatures(degree=2),LinearRegression())
model = make_pipeline(StandardScaler(),LinearRegression())

model.fit(x_train,y_train)
pred = model.predict(x_test)
print(model.score(x_test,y_test))
print(xp.shape)
x_train, x_test, y_train, y_test = train_test_split(xp,y,train_size=0.8,random_state=234,)

model = make_pipeline(PolynomialFeatures(degree=2, include_bias=False),LinearRegression())

model.fit(x_train,y_train)
pred = model.predict(x_test)
print(model.score(x_test,y_test))
from sklearn.preprocessing import MinMaxScaler,StandardScaler,PolynomialFeatures
pf = PolynomialFeatures(degree=2,include_bias=False)
x = pf.fit_transform(x)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x_train, y_train, cv=5, scoring ='r2')
print(np.mean(scores))