import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1.데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split, KFold
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8,random_state=234)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델구성
from sklearn.svm import LinearSVC,SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline, Pipeline
model = Pipeline([('minmax',MinMaxScaler()),('RF',RandomForestClassifier())])
# model = make_pipeline(MinMaxScaler(),PCA(),RandomForestClassifier())
#4개의 칼럼을 스케일러로 조정하고 PCA로 2개로 압축한 다음 RF로 분류했다
#3.훈ㄹ련
model.fit(x_train, y_train)

#4평가예측

result = model.score(x_test,y_test)
print(result)