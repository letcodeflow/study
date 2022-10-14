#1. 데이터
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score,KFold,cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')
pd.__version__
path = 'c:/study/_data/ddareunge/'
train_set = pd.read_csv(path + 'train.csv', index_col=0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)


train_set = train_set.fillna(0) # 결측치 0으로 채움

x = train_set.drop(['count'], axis=1)
y = train_set['count']


x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.8)

allAlgoithms = all_estimators(type_filter='regressor')

from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = KFold(n_splits=n_splits,random_state=2371,shuffle=True)
for (name,algori) in allAlgoithms:
    try:
        model = algori()
        # model.fit(x_train, y_train)
        # y_predict = model.predict(x_test)
        # r2 = r2_score(y_test, y_predict)
        ypred = cross_val_predict(model, x_test, y_test, cv=kfold)

        score = cross_val_score(model,x_train,y_train,cv=kfold)
        p = round(np.mean(score),4)
        print(name, '\n','\n',p)
    except:
        print(name,'안나옴')