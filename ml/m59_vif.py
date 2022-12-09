#다중공선성
#높으면 나쁘다
# import numpy as np
# import pandas as pd

# aaa = []

# for i in range(10):
#     print(i)
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

datasets = fetch_california_housing()
# print(datasets.feature_names) ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

x = datasets.data
y = datasets.target

# print(x.shape, y.shape)(20640, 8) (20640,)

# print(type(x)) <class 'numpy.ndarray'>
x = pd.DataFrame(x, columns=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
)
pd.set_option('display.max_columns',None)
# print(x)
vif = pd.DataFrame()
vif['VIF Factor'] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
vif['features'] = x.columns
# print(vif)
#    VIF Factor    features
# 0   11.511140      MedInc
# 1    7.195917    HouseAge
# 2   45.993601    AveRooms
# 3   43.590314   AveBedrms
# 4    2.935745  Population
# 5    1.095243    AveOccup
# 6  559.874071    Latitude
# 7  633.711654   Longitude
# 10이상이면(혹은 5) 다중공선성이 높다고 판단한다

drop_features = ['Longitude']
# x = x.drop(drop_features, axis=1)
vif = pd.DataFrame()
vif['VIF Factor'] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
vif['features'] = x.columns
# print(vif)
#    VIF Factor    features
# 0    9.865861      MedInc
# 1    6.880512    HouseAge
# 2   42.192223    AveRooms
# 3   39.768396   AveBedrms
# 4    2.793169  Population
# 5    1.094908    AveOccup
# 6   22.498755    Latitude

drop_features = ['AveRooms']
# x = x.drop(drop_features, axis=1)
vif = pd.DataFrame()
vif['VIF Factor'] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
vif['features'] = x.columns
#    VIF Factor    features
# 0    5.036342      MedInc
# 1    6.660725    HouseAge
# 2    6.430073   AveBedrms
# 3    2.752356  Population
# 4    1.094801    AveOccup
# 5   21.922515    Latitude


drop_features = ['Latitude']
# x = x.drop(drop_features, axis=1)
vif = pd.DataFrame()
vif['VIF Factor'] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
vif['features'] = x.columns
print(vif)
#    VIF Factor    features
# 0    3.801038      MedInc
# 1    3.769898    HouseAge
# 2    4.263506   AveBedrms
# 3    2.222516  Population
# 4    1.094644    AveOccup

x_tr, x_test, y_tr, y_test = train_test_split(x,y, random_state=23, test_size=0.2)

model = RandomForestRegressor(n_jobs=-1)

model.fit(x_tr, y_tr)

print(model.score(x_test, y_test))
# after drop 0.660093671169624
# 0.732382091557429
# 0.7400712029510508
# 0.8111846132999971