from sklearn import metrics
from sklearn.datasets import load_boston

data_sets = load_boston()

x = data_sets.data
y = data_sets.target

from sklearn.model_selection import train_test_split,cross_val_score,KFold,StratifiedKFold
import warnings
warnings.filterwarnings('ignore')
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=False)
from sklearn.utils import all_estimators
allAlgorithms = all_estimators(type_filter='regressor')
n_splits=5
kfold = KFold(n_splits=n_splits,shuffle=True, random_state=2398)
# kfold = StratifiedKFold(n_splits=n_splits,shuffle=True, random_state=2398)
from sklearn.metrics import r2_score
import numpy as np
for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        model.fit(x_train, y_train)


        y_predict = model.predict(x_test)

        r2= r2_score(y_test, y_predict)
        p = round(np.mean(cross_val_score(model,x_train,y_train, cv=kfold)),4)
        print(name,'\n',r2,'\n',p)
    except:
        print(name,'nonon')