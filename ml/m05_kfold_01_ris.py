import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
#1.데이터
from sklearn.svm import LinearSVC, SVC #리니어 서포트 벡터 
from sklearn.linear_model import Perceptron,LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score,StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')
datasets = load_iris()
x = datasets['data']
y = datasets['target']
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, shuffle=True, random_state=137)

allAlgorithms = all_estimators(type_filter= 'classifier')
n_splits = 5
# kfold = KFold(n_splits=n_splits,shuffle=True,random_state=66)
kfold = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=66)

for (name,algorithm) in allAlgorithms:
    try:
        model = algorithm()
        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test, y_predict)
        scores = cross_val_score(model,x,y,cv=kfold)
        p = round(np.mean(scores),4)
        print(name, '정답률', acc,'cross_val 평균',p)
    except:
        print(name,'안나옴')
# print('acc',scores,'\n cross_val_score', round(np.mean(scores),4))






