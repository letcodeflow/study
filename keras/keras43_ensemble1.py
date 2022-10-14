#1.데이터
import numpy as np
x1 = np.array([range(100), range(301,401)]) #삼성전나, 하이닉스 종가
x2 = np.array([range(101,201), range(411,511), range(100,200)]) #원유 돈육 밀
x1 = np.transpose(x1)
x2 = np.transpose(x2)

print(x1.shape, x2.shape)

y = np.array(range(2001,2101)) #금리

print(y)
print(y.shape)

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y1_train, y1_test = train_test_split(x1,x2,y, train_size=0.7,random_state = 23
                                                        )
print(x1_train.shape, x1_test.shape, x2_train.shape, x2_test.shape, y1_train.shape, y1_test.shape)

from keras.models import Model
from keras.layers import Dense, Input,Reshape,Conv1D,Flatten
from keras.layers import concatenate,Concatenate

#2. 모델1
input1 = Input(shape=(2,))
den1 = Dense(100, activation='relu', name='den1')(input1)
den2 = Reshape((50,2))(den1)
den3 = Conv1D(10,4, activation='relu', name='den3')(den2)
flat = Flatten()(den3)
den4 = Dense(500, name='soo1')(flat)
#모델2

input11 = Input(shape=(3,))
den11 = Dense(5, activation='relu', name='den11')(input11)
den21 = Dense(5, activation='relu', name='den21')(den11)
den31 = Dense(5, activation='relu', name='den31')(den21)
den41 = Dense(5, activation='relu', name='soo11')(den31)

#concat 엮다
merge1 = concatenate((den4, den41)) #리스트 append 개념 = 하나의 레이어
merge3 = Dense(10,activation='relu', name= 'merg1')(merge1)
merge2 = Dense(10, name= 'merg2')(merge1)
output3 = Dense(20, name= 'merg3')(merge3)

model = Model(inputs=[input11, input1], outputs=output3)
model.summary()


#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit([x2_train,x1_train],y1_train, epochs=100, batch_size=2000)

#4.평가 예측
loss = model.evaluate([x2_test,x1_test],y1_test)
print(loss)

# [2732018.75, 1412.19140625]