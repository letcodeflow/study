#r2 0.62 이상


# from statistics import LinearRegression
from sklearn.datasets import load_diabetes
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x)
print(y)
#y는 어차피 결과값으로 나오는 것이기 때문에 여기서 전처리는 필요없다

print(x.shape)
print(y.shape)

print(datasets.feature_names)
print(datasets.DESCR)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)
from sklearn.svm import SVR, LinearSVR, SVC
from sklearn.linear_model import Perceptron, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
model = LinearSVR()
# model = SVR()
# model = Perceptron()
# model = LinearRegression()
# model = KNeighborsRegressor()
# model = DecisionTreeRegressor()
# model = RandomForestRegressor()

# RANDOM
# r2 0.3525458089804213
# 0.3525458089804213

# decision
# r2 -0.1910228660698292
# -0.1910228660698292

# kneightbor
# r2 0.3968391279034368
# 0.3968391279034368

# linearregre
# r2 0.50638715461148
# 0.50638715461148

# svr
# r2 0.14331393731412367
# 0.14331393731412367

# lineasvr
# r2 -0.33470257008497795
# -0.33470257008497795












model.fit(x_train, y_train)

# #2.모델

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

# model=Sequential()
# model.add(Dense(4, input_dim=10))
# model.add(Dense(100))
# model.add(Dense(1))

#3. 컴파일 훈련
# model.compile(loss = 'mse', optimizer='adam')
# model.fit(x_train, y_train, epochs=200, batch_size=100)

#4. 평가, 예측
# loss = model.evaluate(x_test, y_test)
# print('loss', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2=r2_score(y_test, y_predict)
print('r2', r2)
# print(y_test.shape, y_predict.shape)
# y_test = y_test.reshape(-1,1)
# y_predict = y_predict.reshape(-1,1)
print(model.score(x_test,y_test))

