#1. 데이터
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score,KFold,cross_val_predict,GridSearchCV,RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')
pd.__version__
path = 'c:/study/_data/ddareunge/'
train_set = pd.read_csv(path + 'train.csv', index_col=0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)

print(train_set.isnull().sum())
print(train_set.isna().any()[lambda x:x])
# train_set.fillna(0,inplace=True)
print(train_set.info())
print(train_set.describe())
train_set.dropna(inplace=True)
x = train_set.drop(['count'], axis=1)
y = train_set['count']
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer,KNNImputer,SimpleImputer
imp = IterativeImputer(n_nearest_features=80)
# imp = KNNImputer(n_neighbors=80)
# imp = SimpleImputer(strategy='most_frequent')
train_set = imp.fit_transform(train_set)

def outliers(data_out):
    quartile_1,q2,quartile_3 = np.percentile(data_out,[25,50,75])
    print('1사분위:',quartile_1)
    print('q2:',q2)
    print('3사분위:',quartile_3)
    iqr = quartile_3 -quartile_1
    print('iqr:',iqr)
    lower_bound = quartile_1 -(iqr*1.5)
    upper_bound = quartile_3 +(iqr*1.5)
    return np.where((data_out>upper_bound)|(data_out<lower_bound))

    
print(np.unique(train_set,return_counts=True))
outliers_loc = outliers(train_set[:,6])
print('이상치 위치',outliers_loc)
import matplotlib.pyplot as plt
plt.boxplot(train_set[:,6])
plt.show()





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

print(model1.score(x_test,y_test))
print(model2.score(x_test,y_test))
print(model3.score(x_test,y_test))
print(model4.score(x_test,y_test))

# train_set.fillna(0,inplace=True)
# 0.5816508231075412
# 0.7725623930058763
# 0.7643153182712669
# 0.7670536232913103

# train_set.dropna(inplace=True)
# imp = SimpleImputer(strategy='mean')

# 0.6439193830395543
# 0.7893966914414856
# 0.792978931918626
# 0.8010870525789373