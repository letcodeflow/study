from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
x, y = data.data,data.target

print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=234,train_size=0.8,stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
model = BaggingClassifier(LogisticRegression(),
                          n_estimators=100,
                          n_jobs=-1,
                          random_state=328)

model.fit(x_train,y_train)
print(model.score(x_test,y_test))                    