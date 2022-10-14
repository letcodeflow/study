import warnings
from sklearn.utils import all_estimators
#1.데이터
from sklearn.datasets import fetch_california_housing
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target


from sklearn.model_selection import train_test_split,cross_val_score,KFold
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=66)
#스케일링팥
import numpy as np
from sklearn.metrics import r2_score

allAlgorithms = all_estimators(type_filter='regressor')
n_splits=5
kfold = KFold(random_state=2387,shuffle=True,n_splits=n_splits)
for (name, algori) in allAlgorithms:
    try:
        model = algori()
        model.fit(x_train,y_train)
        y_predict = model.predict(x_test)
        r2=r2_score(y_test, y_predict)
        p = round(np.mean(cross_val_score(model,x_train,y_train,cv=kfold)),4)
        print(name,'\n',r2,'\n',p)
    except:
        print(name,'안돼요')