#데이터 불러오기

from this import d
import pandas as pd
path = './_data/kaggle_bike/'
train_set = pd.read_csv(path + 'train.csv')
test_set = pd.read_csv(path + 'test.csv')

train_set=train_set.dropna()
test_set=test_set.dropna()

x = train_set.drop(['count', 'casual', 'windspeed', 'datetime'], axis=1)
y = train_set['count']

print(x.shape)
print(y.shape)

print(x.info())



#와꾸나옴 트레인셋 테스트셋 분리

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, shuffle=False)

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

model = Sequential()
model.add(Dense(5,input_dim=8))
model.add(Dense(5, activation = 'sigmoid'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(5, activation = 'sigmoid'))
model.add(Dense(50, activation = 'sigmoid'))
model.add(Dense(1, activation = 'sigmoid'))

#컴파일 훈련
model.compile(loss = 'mae', optimizer='adam', metrics=['accuracy', 'mse'])
from tensorflow.python.keras.callbacks import EarlyStopping
ES = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose =1, patience = 20, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs = 100, verbose = 3, 
                #  validation_split=0.2,
                  callbacks=[ES],)

#평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)

print('loss', loss)
print('r2', r2)
#그리기, 프린트
import matplotlib.pyplot as plt
plt.figure(figsize=(9,3))
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.xlabel('xlabel')
plt.ylabel('ylabel')
plt.grid()
plt.title('adkjw')
plt.legend(loc ='lower center')
plt.show()
print('loss', loss)
print('r2', r2)

# 적용
# loss [266.28106689453125, 0.0006123698549345136, 118801.46875]
# r2 -1.4804121062082132

# 미적용
# loss [266.28106689453125, 0.0006123698549345136, 118801.46875]
# r2 -1.4804117832325319

# 차이가 있다고 말하기 어렵다