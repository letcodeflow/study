import numpy as np
import pandas as pd
from sklearn.datasets import load_boston, load_breast_cancer,fetch_covtype,load_wine,load_iris,load_digits
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
#1. 데이터 lda  디폴트시 줄어드는 컬럼
datasets = fetch_covtype() #581012,54 -> 6
datasets = load_breast_cancer() #569,30 ->569,1
datasets = load_digits() #64 -> 9
datasets = load_wine() #178,13 ->178,2
datasets = load_iris() #150,4 ->150,2
import xgboost as xg
print(xg.__version__)
x = datasets.data
y = datasets.target
print(np.unique(y))

print(x.shape) 
lda = LinearDiscriminantAnalysis()
lda.fit(x,y)
x = lda.transform(x)
print(x.shape) 

lda_EVR = lda.explained_variance_ratio_
cumsum = np.cumsum(lda_EVR)
print(cumsum)