from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.datasets import load_wine
import numpy as np
from sqlalchemy import all_
import tensorflow as tf
import warnings
from sklearn.utils import all_estimators
warnings.filterwarnings('ignore')
tf.random.set_seed(137)
# 1.데이터
datasets = load_wine()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True, random_state=137)

allAlgorithm = all_estimators(type_filter='classifier')
n_slpits=5
kfold = KFold(n_splits=n_slpits,shuffle=True,random_state=123)

for (name, algorithm) in allAlgorithm:
    aaa = []
    try:
        model = algorithm()
        # model.fit(x_train,y_train)
        # y_predict = model.predict(x_test)
        # acc = accuracy_score(y_test, y_predict)
        scores = cross_val_score(model,x_train,y_train,cv=kfold)
        p = round(np.mean(scores),4)
        print(name,'\n','\n',p)
    except:
        print(name)
        
# val_score 안에 fold가 있기때문에