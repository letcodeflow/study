import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier,VotingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.metrics import accuracy_score,r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
#1.데이터
dataset = load_breast_cancer()
x, y = dataset.data,dataset.target
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=234,train_size=0.8,stratify=y)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(df.head(7))
#2.모델
lr = LogisticRegression()
knn = KNeighborsClassifier(n_neighbors=8)
xg = XGBClassifier()
lg = LGBMClassifier()
cat = CatBoostClassifier(verbose=0)


model = VotingClassifier(
    estimators=[('XG',xg), ('LG',lg),('CAT',cat)],
    voting= 'soft'
)
#3.훈련
model.fit(x_train,y_train)

#4.평가 예측
pred = model.predict(x_test)
print(accuracy_score(y_test,pred))

# 0.9736842105263158

classifier = [lg,cat,xg]
for model2 in classifier:
    model2.fit(x_train,y_train)
    pred = model2.predict(x_test)
    score2 = accuracy_score(y_test, pred)
    class_name = model2.__class__.__name__
    print('{0} 정확도: {1:.4f}'.format(class_name, score2))
    
# 0.9736842105263158
# LogisticRegression 정확도: 0.9825
# KNeighborsClassifier 정확도: 0.9825

# voting
# 0.9473684210526315
# LGBMClassifier 정확도: 0.9737
# CatBoostClassifier 정확도: 0.9561
# XGBClassifier 정확도: 0.9474