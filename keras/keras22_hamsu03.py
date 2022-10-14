#r2 0.62 이상
import pandas as pd

path = './_data/kaggle_house/'
wholl_train = pd.read_csv(path + 'train.csv')
wholl_test = pd.read_csv(path + 'test.csv')

print(wholl_train.info())
#non-null값 데이터유형 
print(wholl_train.describe())
#평균 등 수치
print(wholl_train.isnull().sum())



x = wholl_train.drop(['Street', 'Id', 'LotShape'], axis=1)
y = wholl_train['Alley']
print(x.head())
print(x.shape)
print(y.shape)

print(x)
print(y)
#y는 어차피 결과값으로 나오는 것이기 때문에 여기서 전처리는 필요없다

print(x.shape)
print(y.shape)

print(x.columns)







from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)

#object값을 불러와서 리스트로 지정한 다음 드롭시켜야함
#스케일링
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import numpy as np
scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_train))
print(np.max(x_train))
print(np.min(x_test))
print(np.max(x_test))





#2.모델

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

model=Sequential()
model.add(Dense(98, input_dim=78))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))
model.summary()




#3. 컴파일 훈련
model.compile(loss = 'mse', optimizer='adam')
from tensorflow.python.keras.callbacks import EarlyStopping
ES = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)
g = model.fit(x_train, y_train, epochs=20, batch_size=100, validation_split=0.1, callbacks=[ES])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss', loss)

y_predict = model.predict(wholl_test)

from sklearn.metrics import r2_score
r2=r2_score(y_test, y_predict)
print('r2', r2)

import matplotlib.pyplot as plt

from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

plt.figure(figsize=(9,6)) #칸만들기
plt.plot(g.history['loss'], marker=',', c='red', label='loss')
#그림그릴거야
plt.plot(g.history['val_loss'], marker=',', c='blue', label='val_loss')
plt.grid()
plt.title('우결')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right') #라벨값위치 생략시 빈자리에 생성
plt.show()
#vali 적용전
# loss 3717.52490234375
# 3/3 [==============================] - 0s 836us/step
# r2 0.4271954209391574

# loss 3948.56884765625
# 3/3 [==============================] - 0s 999us/step
# r2 0.39159565035296795

# loss 3750.03271484375
# 3/3 [==============================] - 0s 500us/step
# r2 0.42218665427150825

#적용후
# loss 4267.72607421875
# 3/3 [==============================] - 0s 499us/step
# r2 0.3424191315644556

# loss 4283.685546875
# 3/3 [==============================] - 0s 997us/step
# r2 0.3399601997549837

# loss 4028.433349609375
# 3/3 [==============================] - 0s 998us/step
# r2 0.3792900271359142






