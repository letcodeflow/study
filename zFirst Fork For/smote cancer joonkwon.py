import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.preprocessing import StandardScaler,MinMaxScaler 
from xgboost import XGBClassifier,XGBRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#1. 데이터
path = 'C:\_data\wine/' # ".은 현재 폴더"
train_set = pd.read_csv('https://github.com/letcodesing/study/raw/main/_data/winequality-white.csv',
                        sep=';',index_col=None,header=0)
# print(train_set.describe())

# print(train_set.isnull().sum()) #(4898, 12)
def outliers(data_out):
    quartile_1, q2 , quartile_3 = np.percentile(data_out,
                                               [25,50,75]) # percentile 백분위
    # print("1사분위 : ",quartile_1) # 25% 위치인수를 기점으로 사이에 값을 구함
    # print("q2 : ",q2) # 50% median과 동일 
    # print("3사분위 : ",quartile_3) # 75% 위치인수를 기점으로 사이에 값을 구함
    iqr =quartile_3-quartile_1  # 75% -25%
    # print("iqr :" ,iqr)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound)|
                    (data_out<lower_bound))
    
    
    
fixed_acidity_out_index= outliers(train_set['fixed acidity'])[0]
print(fixed_acidity_out_index)
print(fixed_acidity_out_index.shape)
volatile_acidity_out_index= outliers(train_set['volatile acidity'])[0]
print(volatile_acidity_out_index)
print(volatile_acidity_out_index.shape)
lead_outlier_index = np.concatenate((fixed_acidity_out_index,volatile_acidity_out_index),axis=None)
print(lead_outlier_index)
print(lead_outlier_index.shape)
print(len(lead_outlier_index))


fixed_acidity_out_index= outliers(train_set['fixed acidity'])[0]
volatile_acidity_out_index= outliers(train_set['volatile acidity'])[0]
citric_acid_out_index= outliers(train_set['citric acid'])[0]
residual_sugar_out_index= outliers(train_set['residual sugar'])[0]
chlorides_out_index= outliers(train_set['chlorides'])[0]
free_sulfur_dioxide_out_index= outliers(train_set['free sulfur dioxide'])[0]
total_sulfur_dioxide_out_index= outliers(train_set['total sulfur dioxide'])[0]
density_out_index= outliers(train_set['density'])[0]
pH_out_index= outliers(train_set['pH'])[0]
sulphates_out_index= outliers(train_set['sulphates'])[0]
alcohol_out_index= outliers(train_set['alcohol'])[0]
# quality_out_index= outliers(train_set['quality'])[0]


lead_outlier_index = np.concatenate((fixed_acidity_out_index,
                                    #  volatile_acidity_out_index,
                                     citric_acid_out_index,
                                     residual_sugar_out_index,
                                    #  chlorides_out_index,
                                     free_sulfur_dioxide_out_index,
                                     total_sulfur_dioxide_out_index,
                                    #  density_out_index,
                                     pH_out_index,
                                     sulphates_out_index,
                                     alcohol_out_index,
                                    #  quality_out_index
                                     ),axis=None)
print(len(lead_outlier_index)) #200
print(lead_outlier_index)