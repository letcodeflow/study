#1.데이터
import numpy as np
x1 = np.array([range(100), range(301,401)]) #삼성전나, 하이닉스 종가
x2 = np.array([range(101,201), range(411,511), range(100,200)]) #원유 돈육 밀
x3 = np.array([range(100,200), range(1301,1401)]) #삼성전나, 하이닉스 종가
x1 = np.transpose(x1)
x2 = np.transpose(x2)
x3 = np.transpose(x3)

print(x1.shape, x2.shape)

y1 = np.array(range(2001,2101)) #금리
y2 = np.array(range(201,301)) #환율

# print(y)
# print(y.shape)

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y1_train,\
y1_test, y2_train, y2_test = train_test_split(x1,x2,x3,y1,y2, train_size=0.7,random_state = 23
                                                        )
print(x1_train.shape, x1_test.shape, x2_train.shape, x2_test.shape,  x3_train.shape, x3_test.shape, y1_train.shape, y1_test.shape, y2_train.shape, y2_test.shape)
print(y1_test)
from keras.models import Model
from keras.layers import Dense, Input,Reshape,Conv1D,Flatten,Conv2D,GRU

#2. 모델1
input1 = Input(shape=(2,))
den1 = Dense(100, activation='relu', name='den1')(input1)
den2 = Reshape((50,2))(den1)
den3 = Conv1D(10,4, activation='relu', name='den3')(den2)
flat = Flatten()(den3)
den4 = Dense(5, name='soo1')(flat)
#모델2

input11 = Input(shape=(3,))
den41 = Dense(8, activation='relu', name='soo11')(input11)
den42 = Dense(8, activation='relu', name='soo12')(den41)
den43 = Dense(8, activation='relu', name='soo13')(den42)
den44 = Dense(8, activation='relu', name='soo14')(den43)
# res1 = Reshape((2,8))
# den11 = Conv2D(20,kernel_size=(1,1), activation='relu', name='den11')(res1)
# den21 = Reshape((2,10))(den11)
# den31 = Conv1D(10,1, activation='relu', name='den32')(res1)
# den31 = GRU(80, activation='relu', name='den31')(den41)

#모델3
input12 = Input(shape=(2,))
den12 = Dense(100, activation='relu', name='den12')(input12)
den22 = Reshape((50,2))(den12)
den32 = Conv1D(10,4, activation='relu', name='den32')(den22)
flat2 = Flatten()(den32)
den42 = Dense(5, name='soo12')(flat2)

from keras.layers import concatenate,Concatenate
#concat 엮다
# merge1 = concatenate((den4, den41, den42)) #리스트 append 개념 = 하나의 레이어
merge1 = Concatenate(1)([den4,den41,den42])
merge3 = Dense(10,activation='relu', name= 'merg1')(merge1)
merge2 = Dense(10, name= 'merg2')(merge3)
output3 = Dense(1, name= 'merg3')(merge2)

# merge11 = concatenate((den4, den41, den42)) #리스트 append 개념 = 하나의 레이어
merge11 = Concatenate(1)([den4,den41,den42])
merge31 = Dense(10,activation='relu', name= 'merg11')(merge11)
merge21 = Dense(10, name= 'merg21')(merge31)
output31 = Dense(1, name= 'merg31')(merge21)

model = Model(inputs=[input11, input1, input12], outputs=[output3, output31])
model.summary()


#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit([x2_train,x1_train,x3_train],[y1_train,y2_train], epochs=100, batch_size=2000)

#4.평가 예측
loss = model.evaluate([x2_test,x1_test,x3_test],[y1_test,y2_test])
loss2 = model.evaluate([x2_test,x1_test,x3_test],y1_test)
loss3 = model.evaluate([x2_test,x1_test,x3_test],y2_test)

print(loss)
print(loss2)
print(loss3)

y_predict1, y_predict2 = model.predict([x2_test,x1_test,x3_test])

print(y_predict1.shape, y_predict2.shape)

from sklearn.metrics import r2_score, mean_squared_error
print(y_predict1)
print(y_predict2)
r2 = r2_score(y1_test, y_predict1)
r22 = r2_score(y2_test, y_predict2)
print(r2)
print(r22)

def RMSE(a,b):
    return np.sqrt(mean_squared_error(a,b))

rmse = RMSE(y1_test, y_predict1)
rmse2 = RMSE(y2_test, y_predict2)
rmse4 = RMSE(y2_test, y_predict1)

print(rmse)
print(rmse2)

print((r2+r22)/2)
print((rmse2+rmse)/2)
# [2732018.75, 1412.19140625]
# [16678.576171875, 96.81687927246094]
# [6380.90185546875, 5434.7158203125, 946.1862182617188, 56.015098571777344, 24.4196720123291]

# [6690.76611328125, 6078.09326171875, 612.6729736328125]