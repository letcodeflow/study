 #데이터
import pandas as pd

path = './_data/kaggle_house/'
train_set = pd.read_csv(path + 'train.csv', index_col=0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'sample_submission.csv')
 
print(train_set.info())
print(train_set.dtypes[train_set.dtypes == "object"].index)
category = train_set.dtypes[train_set.dtypes == "object"].index
train_set = train_set.drop(category, axis=1)
test_set = test_set.drop(category, axis=1)

# train_set.dropna()
# test_set.dropna()
train_set = train_set.fillna(0)
test_set = test_set.fillna(0)


print(train_set.info())
print(train_set.shape)
print(test_set.shape)
print(train_set)
print(test_set)

#모델구서잉 아니고 x,y셋
x = train_set.drop(['SalePrice'], axis=1)
y = train_set['SalePrice']

print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=False)

print(x_train)
print(x_test)
print(y_train)
print(y_test)

#모델구성
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
model = Sequential()
model.add(Dense(5, input_dim=36))
model.add(Dense(5, activation='sigmoid'))
model.add(Dense(5, activation='sigmoid'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#컴파일 훈련
model.compile(loss = 'mae', optimizer = 'adam', metrics=['accuracy', 'mse'])
from tensorflow.python.keras.callbacks import EarlyStopping
ES = EarlyStopping(monitor = 'val_loss', mode='min', patience = 20, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs = 10, verbose=2, 
                 validation_split=0.2,
                  callbacks=[ES]
                 )

#평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
# from sklearn.metrics import r2_score
# r2 = r2_score(y_test, y_predict)
print(y_test.shape, y_predict.shape)
pd.set_option('display.max_rows', None)
print(y_predict)

#그리기 집어넣기 r2값
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.xlabel('loss')
plt.ylabel('epoch')
plt.grid()
plt.title('deep')
plt.legend(loc='center')
# plt.show()
# print('r2', r2)

y_summit = model.predict(test_set)

print(y_summit.shape, submission['Id'].shape, test_set.shape)
print(y_summit)
submission['SalePrice'] = y_summit
submission.to_csv(path + 'submission.csv', index=True)

# [[0.8555535  0.9634037  0.95638365 0.96676093 0.9461465 ]
#  [0.8555535  0.9634037  0.95638365 0.96676093 0.9461465 ]
#  [0.8555535  0.9634037  0.95638365 0.96676093 0.9461465 ]
#  ...
#  [0.8555535  0.9634037  0.95638365 0.96676093 0.9461465 ]
#  [0.8555535  0.9634037  0.95638365 0.96676093 0.9461465 ]
#  [0.8555535  0.9634037  0.95638365 0.96676093 0.9461465 ]]
# 다섯줄의 중복된 값이 나옴
