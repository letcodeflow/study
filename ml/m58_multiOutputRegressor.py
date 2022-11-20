import numpy as np
from sklearn.datasets import load_linnerud
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

x, y = load_linnerud(return_X_y=True)
print(x)
print(y)
print(x.shape, y.shape)

model = Ridge()
model.fit(x,y)
print(model.predict([[2,1,13]])) #[[208.27686672  40.6477907   51.70492405]]

model = XGBRegressor()
model.fit(x,y)
print(model.predict([[2,1,13]])) #[[161.23384   36.867706  56.28511 ]]

model = CatBoostRegressor()
# model.fit(x,y) #Currently only multi-regression, multilabel and survival objectives work with multidimensional target
model = MultiOutputRegressor(CatBoostRegressor())
model.fit(x,y)
print(model.predict([[2,1,13]])) 

model = LGBMRegressor()
model = MultiOutputRegressor(model)
model.fit(x,y)
print(model.predict([[2,1,13]]))