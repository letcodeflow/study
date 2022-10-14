#1 트레인 0.7
#2 r2 0.8이상







#1.데이터

from sklearn.datasets import load_boston


datasets = load_boston()

x = datasets.data
y = datasets.target

print(x.shape)
print(y.shape)
#와꾸를 알아야하니까

# (506, 13)
# (506,) 모델 돌릴때는 (506, )이나 (506,1) 이나 같다 

print(datasets.feature_names) #사이킷런 예제만 가능한 명령
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
#  'B' 'LSTAT']
# FutureWarning: Function load_boston is deprecated; `load_boston` is deprecated in 1.0 and will be removed in 1.2.
# The Boston housing prices dataset has an ethical problem. You can refer to
#     the documentation of this function for further details.
#     Alternative datasets include the California housing dataset (i.e.
#     :func:`~sklearn.datasets.fetch_california_housing`) and the Ames housing
#     dataset. You can load the datasets as follows::
        
print(datasets.DESCR)
# Data Set Characteristics:**

#     :Number of Instances: 506

#     :Number of Attributes: 13 numeric/categorical predictive. Median Value (attribute 14) is usually the target.

#     :Attribute Information (in order):
#         - CRIM     per capita crime rate by town
#         - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.      
#         - INDUS    proportion of non-retail business acres per town
#         - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise) 
#         - NOX      nitric oxides concentration (parts per 10 million)
#         - RM       average number of rooms per dwelling
#         - AGE      proportion of owner-occupied units built prior to 1940
#         - DIS      weighted distances to five Boston employment centres
#         - RAD      index of accessibility to radial highways
#         - TAX      full-value property-tax rate per $10,000
#         - PTRATIO  pupil-teacher ratio by town
#         - B        1000(Bk - 0.63)^2 where Bk is the proportion of black people by town  
#         - LSTAT    % lower status of the population
#         - MEDV     Median value of owner-occupied homes in $1000's

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=66)

print(x.shape)
print(y.shape)

model = Sequential()
model.add(Dense(100, input_dim=13))
model.add(Dense(1000))
model.add(Dense(200))
model.add(Dense(1000))
model.add(Dense(200))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(200))
model.add(Dense(1000))
model.add(Dense(100))
model.add(Dense(20))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mae', optimizer='adam')
model.fit(x_train, y_train, epochs=4500, batch_size=30)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss', loss)
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2', r2)

# loss 17.11542320251465
# r2 0.7928343880564431

# loss 16.793380737304688
# r2 0.7967323913245712

# loss 16.592853546142578
# r2 0.7991595975758828