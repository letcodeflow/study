from sklearn.model_selection import train_test_split,cross_val_score,KFold,RandomizedSearchCV
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
import numpy as np
import warnings
import pandas as pd
pd.set_option('display.max_rows', None)
warnings.filterwarnings('ignore')
path = 'C:/Users/aiapalm/OneDrive - KNOU/beat/study/_data/kaggle_titanic/'
train_set = pd.read_csv(path + 'train.csv', index_col=0)
train_set = train_set.dropna()
test_set = pd.read_csv(path + 'test.csv', index_col=0)
test_set = test_set.dropna()
x = train_set.drop(['Survived','Name', 'Ticket', 'Cabin','Sex','Embarked'], axis=1)
y = train_set['Survived']
print(train_set.isna().any()[lambda x:x])
print(test_set.isna().any()[lambda x:x])
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer,KNNImputer,IterativeImputer
imp = SimpleImputer(strategy='most_frequent')
imp = KNNImputer(n_neighbors=4)
imp = IterativeImputer(n_nearest_features=3)
# train_set['Embarked'] = imp.fit_transform(train_set['Embarked'])
print(test_set['Embarked'].isnull().sum())
train_set['Embarked'].fillna('S',inplace=True)
# test_set['Embarked'].fillna('S',inplace=True)
print(test_set['Embarked'].value_counts(dropna=False))
# print(type(train_set),train_set.isnull().sum(), train_set.info(),train_set.describe(),train_set.head())
train_set = pd.get_dummies(train_set, columns=['Embarked','Sex'])
test_set = pd.get_dummies(test_set, columns=['Embarked','Sex'])
# test_set = pd.get_dummies(test_set, axis= 1,columns=['Embarked','Sex'])
train_set.drop(['Ticket','Cabin','Name'],inplace=True,axis=1)
test_set.drop(['Ticket','Cabin','Name'],inplace=True,axis=1)
# print(train_set['Age'].describe())
# train_set['Age'] = imp.fit_transform(train_set['Age'])
train_set['Age'].fillna(train_set['Age'].mean(),inplace=True)
test_set['Age'].fillna(test_set['Age'].mean(),inplace=True)
test_set['Fare'].fillna(test_set['Fare'].mean(),inplace=True)

# print(test_set.isnull().sum())
# print(train_set.dtypes)
# print(train_set.corr())
# import matplotlib.pyplot as plt
# import seaborn as sns
# # sns.set(1.2)
# sns.heatmap(data = train_set.corr(),square=True,annot=True,cbar=True)
# plt.show()



def outliers(data):
    q1,q2,q3 = np.percentile(data,[25,50,75])
    print(q1,q2,q3)
    iqr = q3-q1
    lower_bound = q1 - (iqr*1.5)
    upper_bound = q3 + (iqr*1.5)
    return np.where((data>upper_bound)|(data<lower_bound))
print(outliers(train_set['Age']))

x_train, x_test, y_train, y_test = train_test_split(
    train_set.drop(['Survived'],axis=1), train_set['Survived'], train_size=0.7, shuffle=True, random_state=137)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
model1 = DecisionTreeClassifier()
model2 = RandomForestClassifier()
model3 = GradientBoostingClassifier()
model4 = XGBClassifier()
# model1.fit(x_train, y_train)
# model2.fit(x_train, y_train)
# model3.fit(x_train, y_train)
# model4.fit(x_train, y_train)

# print(model1.score(x_test,y_test))
# print(model2.score(x_test,y_test))
# print(model3.score(x_test,y_test))
# print(model4.score(x_test,y_test))



from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score

from sklearn.ensemble import VotingClassifier,BaggingClassifier
model = BaggingClassifier(XGBClassifier(),n_jobs=-1,n_estimators=100,random_state=232)
model.fit(x_train,y_train)
print(accuracy_score(y_test,model.predict(x_test)))
# 전처리
# 0.8208955223880597
# 0.835820895522388
# 0.832089552238806
# 0.8059701492537313

# train_set.drop(['Ticket','Cabin','Name','Sex','Embarked'],inplace=True,axis=1)
# 0.7090909090909091
# 0.6909090909090909
# 0.6545454545454545
# 0.6545454545454545

# bagging 100 xgb
# 0.6909090909090909