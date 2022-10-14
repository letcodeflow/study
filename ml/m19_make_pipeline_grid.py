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
# pipe = Pipeline([('minmax',MinMaxScaler()),('RF',RandomForestClassifier())],verbose=2)
pipe = make_pipeline(MinMaxScaler(),PCA(),RandomForestClassifier())
#4개의 칼럼을 스케일러로 조정하고 PCA로 2개로 압축한 다음 RF로 분류했다
#3.훈ㄹ련
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,HalvingGridSearchCV,HalvingRandomSearchCV,KFold,StratifiedKFold
parameters = [
     {'randomforestclassifier__n_estimators':[100,200],'randomforestclassifier__max_depth':[1,3,2,5,10]},
     {'randomforestclassifier__max_depth':[6,8,10,12],'randomforestclassifier__min_samples_leaf':[1234,23,41,22]},
     {'randomforestclassifier__min_samples_leaf':[3,5,7,10],'randomforestclassifier__min_samples_split':[1,23,24,2,3,5]},
     {'randomforestclassifier__min_samples_split':[2,3,5,10],'randomforestclassifier__n_estimators':[400,20]},
     {'randomforestclassifier__n_jobs':[-1,2,4],'randomforestclassifier__n_estimators':[159,1278,2345,1234],'randomforestclassifier__min_samples_leaf':[6,1,80],'randomforestclassifier__min_samples_split':[1795,13947,149875,19387]}
]
n_splits = 10
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=328947)

model = GridSearchCV(pipe, parameters,cv=kfold,verbose=1)
model.fit(x_train, y_train)

#4평가예측

result = model.score(x_test,y_test)
print(result)