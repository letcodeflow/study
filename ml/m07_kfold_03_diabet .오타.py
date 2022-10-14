#r2 0.62 이상

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.utils import all_estimators
import warnings
from sklearn.metrics import r2_score
warnings.filterwarnings('ignore')
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x)
print(y)
#y는 어차피 결과값으로 나오는 것이기 때문에 여기서 전처리는 필요없다

# print(x.shape)
# print(y.shape)

# print(datasets.feature_names)
# print(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)
n_splits = 5
allAlgorithms = all_estimators(type_filter='regressor')
kfold = KFold(n_splits=n_splits,random_state=234,shuffle=True)
for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        # model.fit(x_train, y_train)
        # y_predict = model.predict(x_test)

        # r2=r2_score(y_test, y_predict)

        q = cross_val_score(model,x_train, y_train,cv=kfold)
        p = (round(np.mean(q)),4)
        print(name,'\n','\n',p)
    except:
        print(name, '안돠마ㅓ오')