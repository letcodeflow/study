# n_component >0.95 이상
# xgboost, gridsearch 또는 randomsearch

# 27_2 결과를 넘을것

parameters = [
    {'n_estomators':[100,200,300],'learning_rate':[0.1,0.3,0.001,0.01],'max_depth':[4,5,6]},
    {'n_estimators':[90,100,110],'learning_rate':[-1,0.001,0.01],'max_depth':[4,5,6],'colsample_bytree':[0.6,0.9,1]},
    {'n_estimators':[90,110], 'learning_rate':[0.1,0.001,0.5],'max_depth':[4,5,6],'colsample_bytree':[0.6,0.9,1],'colsample_bylevel':[0.6,0.7,0.9]}
]
# n_jobs = -1
#     tree_method = 'gpu_hist',
#     predictor='gpu_predictor',
#     gpu_id = 0,
from keras.datasets import mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()
from sklearn.model_selection import train_test_split, KFold,GridSearchCV,RandomizedSearchCV
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.preprocessing import MinMaxScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
pipe = Pipeline([('minmax',MinMaxScaler()),('RF',RandomForestClassifier())],verbose=2)
pipe = make_pipeline(MinMaxScaler(), RandomForestClassifier(),verbose=2)
n_splits = 5
kfold = KFold(n_splits=n_splits,shuffle=True, random_state=3457)
# model1 = DecisionTreeClassifier(pipe,parameters)
# model2 = RandomForestClassifier(pipe,parameters)
model3 = GradientBoostingClassifier(pipe,parameters,cv=kfold,verbose=2)
model4 = XGBClassifier(pipe,parameters,cv=kfold,verbose=2)

model1.fit(x_train, y_train)
model2.fit(x_train, y_train)
model3.fit(x_train, y_train)
model4.fit(x_train, y_train)


print(model1.score)
print(model2.score)
print(model3.score)
print(model4.score)