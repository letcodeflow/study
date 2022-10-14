import tensorflow as tf
from sklearn.model_selection import train_test_split,cross_val_score,KFold,RandomizedSearchCV
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
import numpy as np
import warnings
import pandas as pd
warnings.filterwarnings('ignore')
path = 'c:/study/_data/kaggle_titanic/'
train_set = pd.read_csv(path + 'train.csv', index_col=0)
train_set = train_set.dropna()
test_set = pd.read_csv(path + 'test.csv', index_col=0)
test_set = test_set.dropna()
x = train_set.drop(['Survived','Name', 'Ticket', 'Cabin','Sex','Embarked'], axis=1)
y = train_set['Survived']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True, random_state=137)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
model1 = DecisionTreeClassifier()
model2 = RandomForestClassifier()
model3 = GradientBoostingClassifier()
model4 = XGBClassifier()
model1.fit(x_train, y_train)
model2.fit(x_train, y_train)
model3.fit(x_train, y_train)
model4.fit(x_train, y_train)

print(model1.feature_importances_)
print(model2.feature_importances_)
print(model3.feature_importances_)
print(model4.feature_importances_)