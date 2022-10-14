from tkinter import Grid
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, train_test_split, KFold,GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler #대문자 클래스 약어도 from sklearn.svm import LinearSVC, SVC #리니어 서포트 벡터 
from sklearn.linear_model import Perceptron,LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')
allAlgorithms = all_estimators(type_filter='classifier')
n_splits = 5
kfold = KFold(n_splits=n_splits, random_state=214,shuffle=True)
#1.데이터
data_sets = load_breast_cancer()

x = data_sets['data']
y = data_sets['target']
print(x.shape)
x = x[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,21,22,23,24,25,26,27,28,29]]
print(x.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import KFold, cross_val_score,GridSearchCV, StratifiedKFold,RandomizedSearchCV,HalvingGridSearchCV,HalvingRandomSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline, Pipeline
pipe = Pipeline([('minmax',MinMaxScaler()),('RF',RandomForestClassifier())],verbose=2)
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,HalvingGridSearchCV,HalvingRandomSearchCV,KFold,StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
pca = PCA(n_components=4)
x = pca.fit_transform(x)
model = DecisionTreeClassifier()
model = RandomForestClassifier()
model = GradientBoostingClassifier()
model = XGBClassifier()
model.fit(x_train, y_train)
print(model.score(x_test,y_test))