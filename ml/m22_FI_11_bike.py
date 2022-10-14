#1.트레인 데이터 불러와서 결측치 확인 및 제거, 트레인과 테스트셋 나누기
import pandas as pd
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')
path = 'c:/study/_data/kaggle_bike/'
train_set = pd.read_csv(path + 'train.csv', index_col=0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)
train_set=train_set.dropna()
x = train_set.drop(['casual', 'registered', 'count'], axis=1)
y = train_set['count']
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score,KFold,GridSearchCV,RandomizedSearchCV

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True, random_state=137)

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
model1 = DecisionTreeRegressor()
model2 = RandomForestRegressor()
model3 = GradientBoostingRegressor()
model4 = XGBRegressor()
model1.fit(x_train, y_train)
model2.fit(x_train, y_train)
model3.fit(x_train, y_train)
model4.fit(x_train, y_train)

print(model1.feature_importances_)
print(model2.feature_importances_)
print(model3.feature_importances_)
print(model4.feature_importances_)