import tensorflow as tf
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
import numpy as np
import warnings
import pandas as pd
warnings.filterwarnings('ignore')
path = 'c:/study/_data/kaggle_titanic/'
train_set = pd.read_csv(path + 'train.csv', index_col=0)
train_set = train_set.dropna()
test_set = pd.read_csv(path + 'test.csv', index_col=0)
test_set = test_set.dropna()
x = train_set.drop(['Survived','Name', 'Ticket', 'Cabin','Sex','Embarked'], axis=1)
y = train_set['Survived']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=137)

n_splits = 5
kfold = KFold(random_state=234,shuffle=True,n_splits=n_splits)
allAlgrithms = all_estimators(type_filter='classifier')
for (name,algorithm) in allAlgrithms:
    try:
        model = algorithm()
        # model.fit(x_train,y_train)
        
        # y_predict = model.predict(x_test)
        # acc = accuracy_score(y_test, y_predict)
        p = (round(np.mean(cross_val_score(model,x_train,y_train,cv=kfold))))
        print(name,'\n','\n',p)
    except:
        print(name,'안나옴')












