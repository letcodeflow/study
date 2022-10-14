from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional
 
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
            [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],
            [20,30,40], [30,40,50], [40,50,60]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

from sklearn.model_selection import train_test_split

# x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, random_state=55)
 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()
# scaler.fit(x_train) # 실행

# x = scaler.transform(x_train) # 전환 evaluate, preidict과정 같은 것
# x = scaler.transform(x_test) # 전환 evaluate, preidict과정 같은 것
print(x.shape) # (13,3)
print(y.shape) # (13,)
x = x.reshape((x.shape[0], x.shape[1], 1))
print(x.shape) # (13,3,1)
 
model = Sequential()
model.add(Bidirectional( LSTM(200,return_sequences=True, activation = 'relu'), input_shape=(3,1)))
model.add(LSTM(200,return_sequences=True, activation = 'relu', ))
model.add(Bidirectional( LSTM(200,return_sequences=False, activation = 'relu', )))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(100, activation='relu'))
model.add(Dense(20, activation='sigmoid'))
model.add(Dense(1))
# model.summary()
model.compile(optimizer='adam', loss='mse')
 
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')
# loss값을 모니터해서 과적합이 생기면 100번 더 돌고 끊음
# mode=auto loss면 최저값이100번정도 반복되면 정지, acc면 최고값이 100번정도 반복되면 정지
# mode=min, mode=max
model.fit(x, y, epochs=1000, batch_size=1, verbose=1, callbacks=[early_stopping])
 
x_input = array([50,60,70]) # predict용
x_input = x_input.reshape((1,3,1))
# 들어가는 x와꾸와 같이 리쉐잎해줌
yhat = model.predict(x_input)
print(yhat)

