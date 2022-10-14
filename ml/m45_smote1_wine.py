import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
from imblearn.over_sampling import SMOTE
import sklearn as sk
print(sk.__version__)

data = load_wine()
x = data.data
y = data.target
print(pd.Series(y).value_counts())

x_new = x[:-40]
y_new = y[:-40]
# y_new = y_new.reshape(-1,1)
print(type(x),type(y))
print(pd.Series(y_new).value_counts())
print(x_new.shape, y_new.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=234,stratify=y)
print(pd.Series(y_train).value_counts())
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train,y_train)
pred = model.predict(x_test)
print(model.score)
from sklearn.metrics import accuracy_score,f1_score
print('acc',accuracy_score(y_test,pred),f1_score(y_test,pred,average='macro'))
print('acc',accuracy_score(y_test,pred),f1_score(y_test,pred,average='micro'))

smote = SMOTE(random_state=45)
smote.fit_resample(x_train,y_train)
print(pd.Series(y_train).value_counts())
model.fit(x_train,y_train)

print(model.score)
from sklearn.metrics import accuracy_score,f1_score
print('acc',accuracy_score(y_test,pred),f1_score(y_test,pred,average='macro'))
print('acc',accuracy_score(y_test,pred),f1_score(y_test,pred,average='micro'))
