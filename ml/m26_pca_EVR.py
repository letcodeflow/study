from random import random
import numpy as np
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)
pca = PCA(n_components=5)
x = pca.fit_transform(x)
print(x.shape)
pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)
print(sum(pca_EVR))
# [8.05823175e-01 1.63051968e-01 2.13486092e-02 6.95699061e-03
#  1.29995193e-03 7.27220158e-04 4.19044539e-04 2.48538539e-04
#  8.53912023e-05 3.08071548e-05 6.65623182e-06 1.56778461e-06]
# 0.9999999203185791
cumsum = np.cumsum(pca_EVR)
print(cumsum)

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()
""" 
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=32)
#2.모델
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
model = RandomForestRegressor()

#3.훈련
model.fit(x_train, y_train,# eval_metric='error')
)

#4.평가 예측
result = model.score(x_test, y_test)
print(result)

# (506, 13)
# 0.8766385690808591 """