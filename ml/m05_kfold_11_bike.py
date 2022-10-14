#1.트레인 데이터 불러와서 결측치 확인 및 제거, 트레인과 테스트셋 나누기
import pandas as pd
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')
path = 'c:/study/_data/kaggle_bike/'
train_set = pd.read_csv(path + 'train.csv', index_col=0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)
train_set=train_set.dropna()
x = train_set.drop(['casual', 'registered', 'count'], axis=1)
y = train_set['count']
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score,KFold

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=False)
allAlgorithm = all_estimators(type_filter='regressor')
n_splits = 5
kfold = KFold(shuffle=True, random_state=2374,n_splits=n_splits)
for (name, algo) in allAlgorithm:
    try:
        model = algo()
        model.fit(x_train,y_train)
        y_predict = model.predict(x_test)
        r2= r2_score(y_test, y_predict)
        p  = round(np.mean(cross_val_score(model,x_train,y_train,cv=kfold)),4)
        print(name,'\n',r2,'\n',p)
    except:
        print(name,'nono')