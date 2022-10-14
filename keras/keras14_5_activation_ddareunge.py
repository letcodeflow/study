#1.데이터 불러오기
#testset, train,set

import pandas as pd
import sklearn
from sklearn.tree import plot_tree
path = './_data/ddareunge/'
train_set = pd.read_csv(path + 'train.csv')
test_set = pd.read_csv(path + 'test.csv')

print(train_set.shape)
print(test_set.shape)
train_set = train_set.dropna()
test_set = test_set.dropna()
x = train_set.drop(['count'], axis=1)
y = train_set['count']
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=False)
#모델구성
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
model= Sequential()
model.add(Dense(5, input_dim=10))
model.add(Dense(5, activation='sigmoid'))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

#컴파일 훈련
model.compile(loss = 'mae', optimizer = 'adam', metrics=['accuracy', 'mse'])

from tensorflow.python.keras.callbacks import EarlyStopping
ES = EarlyStopping(monitor = 'val_loss', patience = 20,
                   verbose=1, mode='min', restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=500, batch_size=20, verbose=2,
          validation_split=0.2,
           callbacks=[ES]
           )

# 평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)

print('loss', loss)
print('r2', r2)

#그리기
import matplotlib.pyplot as plt
plt.figure(figsize=(3,6))
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.xlabel('xlabel')
plt.ylabel('ylabel')
plt.legend(loc='center left')
plt.title('pltplt')
plt.show()

# 미적용
# loss [105.77693939208984, 0.002506265649572015, 17389.48046875]
# r2 -1.8044295129490617

# 적용
# loss [105.7769546508789, 0.002506265649572015, 17389.486328125]
# r2 -1.8044306070095986

# 크게 차이 없었다