import tensorflow as tf
tf.random.set_seed(137)
import numpy as np
from sklearn.datasets  import load_digits
from sklearn.utils import all_estimators
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

#1.데이터
datasets = load_digits()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, shuffle=True, random_state=137)

n_splits = 5
allAlgorithms = all_estimators(type_filter='classifier')
kfold = KFold(random_state=234,shuffle=True, n_splits=n_splits)
for (name, algoithm) in allAlgorithms:
    try:
        model = algoithm
        model.fit(x_train,y_train)
        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test, y_predict)
        p = round(np.mean(cross_val_score(model.x_train,y_train,cv=kfold)),4)
        print(name,'\n',acc,'\n',p)
    except:
        print(name,'\n생략')
        