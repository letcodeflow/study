#1/4 에서 1/5 삭제 후 비교
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1.데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

print(x.shape)
x = x[:,[2,3]]
print(x.shape)
from sklearn.model_selection import train_test_split, KFold
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8,random_state=234)


#2. 모델구성
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
model1 = DecisionTreeClassifier()
model2 = RandomForestClassifier()
model3 = GradientBoostingClassifier()
model4 = XGBClassifier()
model1.fit(x_train, y_train)
model2.fit(x_train, y_train)
model3.fit(x_train, y_train)
model4.fit(x_train, y_train)

print(model1.feature_importances_)
print(model2.feature_importances_)
print(model3.feature_importances_)
print(model4.feature_importances_)
#4평가예측


# import matplotlib.pyplot as plt

# def plot_feature_importances_dataset(model):
#     n_features = datasets.data.shape[1]
#     plt.barh(np.arange(n_features),model.feature_importances_,align='center')
#     plt.yticks(np.arange(n_features),datasets.feature_names)
#     plt.xlabel('feature importances')
#     plt.ylabel('features')
#     plt.ylim(-1,n_features)

# plt.subplot(4,1,1)
# plot_feature_importances_dataset(model1)
# plt.subplot(4,1,2)
# plot_feature_importances_dataset(model2)
# plt.subplot(4,1,3)
# plot_feature_importances_dataset(model3)
# plt.subplot(4,1,4)
# plot_feature_importances_dataset(model4)
# plt.show()