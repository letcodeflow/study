from random import random
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)
pca = PCA(n_components=12)
x = pca.fit_transform(x)
print(x.shape)

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
# 0.8766385690808591