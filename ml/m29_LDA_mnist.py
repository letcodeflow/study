# [실습]
# 784개 DNN으로 만든거(최상의 성능인 거 // 0.978이상)과 비교!!


import numpy as np
from sklearn.decomposition import PCA
from keras.datasets import mnist
from sklearn.svm import LinearSVC, SVC
# 2.8버전부터 tensorflow.keras 안써도 됨, 걍 keras.으로 쓰라고 나옴
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import time
import warnings
warnings.filterwarnings(action='ignore')

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(y_train.shape)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
y_train = y_train.reshape(y_train.shape[0], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
y_test = y_test.reshape(y_test.shape[0], 1)
x = np.append(x_train, x_test, axis=0)
y = np.append(y_train, y_test, axis=0)
print(x.shape)
print(np.unique(y_train))

# from sklearn.preprocessing import LabelEncoder
# lb = LabelEncoder()
# y = lb.fit_transform(y)
a=[]
# pca = PCA(n_components=x_train.shape[1]-80)
# x = pca.fit_transform(x)
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# lda = LinearDiscriminantAnalysis(n_components=8)
# lda.fit(x,y)
# x = lda.transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=688)
model = XGBClassifier(verbose=2,tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=1)
import time
start=time.time()
model.fit(x_train, y_train)
end=time.time()
result = model.score(x_test, y_test)
print('result',result)
print('time',end-start)
    
# x(70000, 784)
# y[0 1 2 3 4 5 6 7 8 9]

# lda 8
# result 0.9130714285714285
# time 4.1180946826934814

# pca 700
# result 0.9602142857142857
# time 24.425469398498535


# no pca lda
# result 0.9795714285714285
# time 24.087063789367676