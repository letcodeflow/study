import numpy as np
import pandas as pd
from sklearn.datasets import load_boston, load_breast_cancer,fetch_covtype,load_wine,load_iris,load_digits
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
#1. 데이터
datasets = fetch_covtype()
datasets = load_breast_cancer()
datasets = load_digits()
datasets = load_wine()
datasets = load_iris()
import xgboost as xg
print(xg.__version__)
x = datasets.data
y = datasets.target
print(x.shape) #(581012, 54)
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# y = le.fit_transform(y)

# pca = PCA(n_components=2)
# x = pca.fit_transform(x)
# lda = LinearDiscriminantAnalysis(n_components = 2)
# lda.fit(x,y)
# x = lda.transform(x)
# pca_EVR = pca.explained_variance_ratio_
# cumsum = np.cumsum(pca_EVR)
# y = pd.get_dummies(y)
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8,random_state=234,stratify=y)
print(np.unique(y_train,return_counts=True)) #(array([1, 2, 3, 4, 5, 6, 7]), array([169472, 226640,  28603,   2198,   7594,  13894,  16408],

#2.모델
from xgboost import XGBClassifier
model = XGBClassifier(verbose=2,tree_method='gpu_hist',predictor='gpu_predictor',gpu_id=0)

#3.훈련
import time
start = time.time()
model.fit(x_train, y_train)
print('시간',time.time()-start)
#4.평가 예측
result = model.score(x_test, y_test)
print('결과',result)

from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
thresholds = model.feature_importances_
print(thresholds)
print(f'====================')
for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh,)
    
    selec_x_train = selection.transform(x_train)
    selec_x_test = selection.transform(x_test)
    print(selec_x_train.shape, selec_x_test.shape)

    model2 = XGBClassifier(n_jobs=-1,random_state=238,n_estimators=1000,
                      learning_rate=0.1,
                      max_depth=3,
                      gemma=1,tree_method='gpu_hist',predictor='gpu_predictor',gpu_id=0)
    
    model2.fit(selec_x_train,y_train)
    y_pred = model2.predict(selec_x_test)
    score = accuracy_score(y_test,y_pred)
    print("thresh=%.3f, n=%d, r2: %.2f%%"
          % (thresh, selec_x_train.shape[1],score*100))

# xxgboost
# 시간 7.320397138595581
# 결과 0.8953985697443267

# no pca
# 시간 5.952320337295532
# 결과 0.8683166527542319

# pca 18
# 시간 4.731928825378418
# 결과 0.8846587437501614

# lda 
# 시간 3.589005708694458
# 결과 0.7874581551250829

# lda compo 2
# 시간 0.7507467269897461
# 결과 1.0

# pca 2
# 시간 0.4982614517211914
# 결과 0.8666666666666667

# no pca lda
# 시간 0.5761358737945557
# 결과 0.9333333333333333