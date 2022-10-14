#1. 데이터
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout,LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import time


pd.__version__
path = 'c:/study/_data/ddareunge/'
train_set = pd.read_csv(path + 'train.csv', index_col=0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)

print(test_set.shape)

#자료구조를 봐야하니까
print(train_set)
print(train_set.shape)
print(train_set.describe())
print(train_set.info())

print('널값이 ', train_set.isnull().sum())
train_set = train_set.dropna()
print('널값이 삭제이후 ', train_set.isnull().sum())
print(train_set.shape)


x = train_set.drop(['count'], axis=1)
print('x컬럼', x.shape)
y = train_set['count'].values.reshape(-1,1)
print('y 컬럼', y.shape)


x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.99, shuffle=False)
from sklearn.preprocessing import MinMaxScaler, StandardScaler #대문자 클래스 약어도 ㄷ대문자로

# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train))
print(np.max(x_train))
print(np.min(x_test))
print(np.max(x_test))

print(x_train)
print(y_train)

#2.모델구성

model = Sequential()
model.add(LSTM(200, input_shape=(3,3)))
model.add(Dense(200))
model.add(Dense(200))
model.add(Dropout(0.3))
model.add(Dense(200))
model.add(Dense(200))
model.add(Dropout(0.3))
model.add(Dense(200))
model.add(Dropout(0.2))
model.add(Dense(200))
model.add(Dense(200))
model.add(Dense(1))
model.summary()

import tensorflow as tf
from sklearn.metrics import accuracy_score
x = tf.compat.v1.placeholder(tf.float32,shape=[None,9])
y = tf.compat.v1.placeholder(tf.float32,shape=[None,1])

w = tf.Variable(tf.random.normal([9,18]),tf.float32)
b = tf.Variable(tf.random.normal([18]),tf.float32)
hidden = tf.matmul(x,w)+b

w2 = tf.Variable(tf.random.normal([18,32]),tf.float32)
b2 = tf.Variable(tf.random.normal([32]),tf.float32)
hidden2 = tf.matmul(hidden,w2)+b2

w3 = tf.Variable(tf.random.normal([32,16]),tf.float32)
b3 = tf.Variable(tf.random.normal([16]),tf.float32)
hidden3 = tf.matmul(hidden2,w3)+b3

w4 = tf.Variable(tf.random.normal([16,1]),tf.float32)
b4 = tf.Variable(tf.random.normal([1]),tf.float32)
hypothesis = tf.nn.softmax(tf.matmul(hidden3,w4)+b4)
loss = tf.reduce_mean(-tf.reduce_mean(y*tf.log(hypothesis),axis=1))
train = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20):
        _ = sess.run(train,feed_dict={x:x_train,y:y_train})
        if i % 2==0:
            print(i)
    pred = sess.run(hypothesis,feed_dict = {x:x_test})
    # pred = np.argmax(pred,axis=1)
    # y_test = np.argmax(y_test,axis=1)
    print(r2_score(y_test,pred))

# loss 2116.8349609375
# 3/3 [==============================] - 0s 4ms/step
# r2 0.7308391739043492
# rmse 46.00907729565719

# loss 2223.976806640625
# 1/1 [==============================] - 0s 94ms/step
# r2 0.8147804410965995
# rmse 47.1590564329388
# 0.383059024810791


# y_summit = model.predict(test_set)
# print(y_summit)
# print(y_summit.shape) #(715, 1)

# ## .to_csv() 로 submission.csv 에 입력

# submission = pd.read_csv(path + 'submission.csv')#함수정의하고 값 불러오기
# submission['count'] = y_summit #카운트에 y_summit 덮어씌우기


# submission.to_csv(path + 'submission.csv') #y_summit이 덮어씌어진 submission을 불러온 파일에 다시 덮어씌우기
# loss 2489.98486328125
# r2 0.7926264243669512
# rmse 49.89975258447158
# 0.3418304920196533

#민맥스
# loss 2868.696533203125
# r2 0.7610862259980802
# rmse 53.5602109213241
# 0.33219313621520996

#스탠
# loss 2199.585693359375
# r2 0.8168118026787894
# rmse 46.899739313256795
# 0.3275487422943115

# dropout
# loss 2461.12255859375
# r2 0.7950301973877011
# rmse 49.609702847852525
# 0.25470495223999023

# lstm
# loss 1639.4351806640625
# r2 0.8634628326163756
# rmse 40.48993957082207