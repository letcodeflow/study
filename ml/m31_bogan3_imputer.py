import pandas as pd
import numpy as np

data = pd.DataFrame([[2,np.nan,6,8,10],
                     [2,4,np.nan,8,np.nan],
                     [2,4,6,8,10],
                     [np.nan,4,np.nan,8,np.nan]])

data = data.T
data.columns = ['a','b','c','d']
print(data)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer,KNNImputer, IterativeImputer
imputer = SimpleImputer()
imputer = SimpleImputer(strategy='mean') #평균전략
imputer = SimpleImputer(strategy='median') #평균전략
imputer = SimpleImputer(strategy='most_frequent') #평균전략
imputer = SimpleImputer(strategy='constant') #상수 default 0
imputer = SimpleImputer(strategy='constant',fill_value=9) #상수 default 0
imputer = KNNImputer()
imputer = KNNImputer(n_neighbors=1) #평균전략
# imputer = KNNImputer(strategy='median') #평균전략
# imputer = KNNImputer(strategy='most_frequent') #평균전략
# imputer = KNNImputer(strategy='constant') #상수 default 0
# imputer = KNNImputer(strategy='constant',fill_value=9) #상수 default 0
# imputer = IterativeImputer()
# imputer = IterativeImputer(strategy='mean') #평균전략
# imputer = IterativeImputer(strategy='median') #평균전략
# imputer = IterativeImputer(strategy='most_frequent') #평균전략
# imputer = IterativeImputer(strategy='constant') #상수 default 0
imputer = IterativeImputer(n_nearest_features=2) #상수 default 0
imputer.fit(data)
data2 = imputer.transform(data)

print(data2)