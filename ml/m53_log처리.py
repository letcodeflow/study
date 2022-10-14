import numpy as np
import pandas as pd
from inspect import Parameter
from sklearn.datasets import load_boston
from sklearn.model_selection import GridSearchCV,KFold,StratifiedKFold,train_test_split
from xgboost import XGBClassifier,XGBRegressor
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,RobustScaler,QuantileTransformer,PowerTransformer
import matplotlib.pyplot as plt

#1.data
datasets = load_boston()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=234,)

scaler = MinMaxScaler()
scaler = StandardScaler()
scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler = QuantileTransformer()
scaler = PowerTransformer()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2.모델
model = LinearRegression()

model.fit(x_train,y_train)

pred = model.predict(x_test)

result = r2_score(y_test,pred)

print(round(result,4))

#컬럼명 확인위해 판다스변화

df = pd.DataFrame(x,columns=[datasets.feature_names])
print(df)

df.plot.box()
plt.title('snl')
plt.xlabel('feature')
plt.ylabel('데닝터')
# plt.show()

print(df['B'])
df['B'] = np.log1p(df['B'])
# df['CRIM'] = np.log1p(df['CRIM'])
df['ZN'] = np.log1p(df['ZN'])
df['TAX'] = np.log1p(df['TAX'])
x_train, x_test, y_train, y_test = train_test_split(df,y,train_size=0.8,random_state=234,)

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler = QuantileTransformer()
# scaler = PowerTransformer()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2.모델
model = LinearRegression()

model.fit(x_train,y_train)

pred = model.predict(x_test)

result = r2_score(y_test,pred)

print(round(result,4))

# 기존 스케일러 linear
# 0.8086

# 그냥
# 0.7791

# b log 후
# 0.7792

# crime log후
# 0.7784

# zn
# 0.7759

# tax
# 0.7784