0import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1.데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split, KFold
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8,random_state=234)


#2. 모델구성
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
model = DecisionTreeClassifier()
model = RandomForestClassifier()
model = GradientBoostingClassifier()
model = XGBClassifier()
parameters = [
     {'RF__n_estimators':[100,200],'RF__max_depth':[1,3,2,5,10]},
     {'RF__max_depth':[6,8,10,12],'RF__min_samples_leaf':[12,23,41,22]},
     {'RF__min_samples_leaf':[3,5,7,10],'RF__min_samples_split':[1,23,24,2,3,5]},
     {'RF__min_samples_split':[2,3,5,10],'RF__n_estimators':[400,20]},
     {'RF__n_jobs':[-1,2,4],'RF__n_estimators':[159,12,23,12],'RF__min_samples_leaf':[6,1,80],'RF__min_samples_split':[17,13,14,19]}
]
n_splits = 10
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=328947)

model = GridSearchCV(pipe, parameters,cv=kfold,verbose=1)
model.fit(x_train, y_train)

#4평가예측

result = model.score(x_test,y_test)
print(result)
print(model.feature_importances_)