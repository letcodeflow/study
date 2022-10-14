import tensorflow as tf
#텐서플로로 웨이트값에 처음 난수를 이렇게 주겠다 데이터값에 주는 난수표와는 다름 

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
#1.데이터
datasets = load_iris()
x = datasets['data']
y = datasets['target'].reshape(-1,1)

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
encoder.fit(y)
y = encoder.transform(y).toarray()
# 베꼈는데 먼소린지 모르겠다. 일단 y를 정의하면서 리쉐잎으로 정렬하고 그것을 사이킷런 원핫엔코더로 fit처리 한다음에 다시 엔코더의 트랜스폼하고 투어레이로 정렬한다
print(x.shape,y.shape)

# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# print(y)
# print(y.shape) #(150, 3)

# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder(sparse=False)
# datasets = ohe.fit_transform(datasets[['Class Correlation']])
# print(datasets)


x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, shuffle=True, random_state=137)
#shuffle=False 일경우 순차데이터로 정렬돼있기때문에 한쪽 특성값이 잘려나가게 된다

# from tensorflow.python.keras import to_categorical
# y_train = to_categorical(y_train, 3)
# y_test = to_categorical(y_test, 3)

# from sklearn.preprocessing import OneHotEncoder
# oh=OneHotEncoder
# y_train = y_train.reshape(-1,1)
# print(y_train)
# y_train = oh.fit(y_train, None)
# print(y_train)
# import pandas as pd
# y_train = pd.get_dummies

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
from tensorflow.python.keras.layers import Dense, Dropout, LSTM
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping
ES = EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True)
model = Sequential()
model.add(LSTM(5, input_shape=(2,2)))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))

x = tf.compat.v1.placeholder(tf.float32,shape=[None,4])
y = tf.compat.v1.placeholder(tf.float32,shape=[None,3])

w = tf.Variable(tf.random.normal([4,20]),tf.float32)
b = tf.Variable(tf.random.normal([20]),tf.float32)
hidden = tf.matmul(x,w)+b

w2 = tf.Variable(tf.random.normal([20,30]),tf.float32)
b2 = tf.Variable(tf.random.normal([30]),tf.float32)
hidden2 = tf.matmul(hidden,w2)+b2

w3 = tf.Variable(tf.random.normal([30,40]),tf.float32)
b3 = tf.Variable(tf.random.normal([40]),tf.float32)
hidden3 = tf.matmul(hidden2,w3)+b3

w4 = tf.Variable(tf.random.normal([40,3]),tf.float32)
b4 = tf.Variable(tf.random.normal([3]),tf.float32)
hypothesis = tf.nn.softmax(tf.matmul(hidden3,w4)+b4)

loss = tf.reduce_mean(-tf.reduce_mean(y*tf.log(hypothesis),axis=1))
train = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss)
from sklearn.metrics import accuracy_score
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20):
        _ = sess.run(train,feed_dict={x:x_train,y:y_train})
        if i % 2 ==0:
            print(i)
    pred = np.argmax(sess.run(hypothesis,feed_dict={x:x_test}),axis=1)
    y_test = np.argmax(y_test,axis=1)
    print(accuracy_score(y_test,pred))

# #3.컴파일 훈련
# model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# hist = model.fit(x_train, y_train, epochs=10, verbose=1, 
#           validation_split=0.2, callbacks=ES)

# #4.평가 예측
# loss = model.evaluate(x_test, y_test)

# y_predict = model.predict(x_test)
# # y_predict = model.predict(x_test[:5])
# print(y_test)
# print(y_predict)

# # [[0. 0. 1.]
# #  [0. 1. 0.]
# #  [0. 0. 1.]
# #  [0. 0. 1.]
# #  [1. 0. 0.]]
# # [[7.8074373e-03 4.7584463e-02 9.4460809e-01]
# #  [4.6893386e-03 9.9143785e-01 3.8727468e-03]
# #  [7.8190332e-03 4.7564059e-02 9.4461691e-01]
# #  [7.8232400e-03 4.7535945e-02 9.4464082e-01]
# #  [9.9765587e-01 2.1136946e-03 2.3034363e-04]]
# #제일 큰값만 1로 나머지 0
# # y_predict = np.argmax(y_predict, axis=3).reshape(-1)
# y_predict = np.argmax(y_predict, axis= 1)
# y_test = np.argmax(y_test, axis= 1)
# print(y_test)
# print(y_predict)

# #predic값이 3개 출력됨 합은 1
# #test값도 3개 출력됨 둘다 (y,) 로 바꿔줌



# print(y_test[:5])







# from sklearn.metrics import accuracy_score


# # y_predict = y_predict.round(0)
# # # pre2 = y_predict.flatten() # 차원 펴주기
# # # pre3 = np.where(pre2 > 1, 2 , 0) #1보다크면 2, 작으면 0
# acc = accuracy_score(y_test, y_predict)

# # print(y_predict)
# print('loss : ', loss[0])
# #loss식의 첫번째
# print('acc :',  loss[1])
# #loss식의 두번째
# print('acc', acc)

# loss [0.3140193819999695, 0.0]
# r2 0.0

# loss [0.46477609872817993, 0.0]
# r2 0.0

# loss :  0.4026905596256256
# acc : 0.0
# acc 0.0

# loss :  0.0
# acc : 0.0
# acc 0.0
# 왜다 0인가

# loss :  0.10120422393083572
# acc : 1.0
#원핫인코딩 사이킷런 코드 복붙하니 나온값. 어떻게 움직이는지는 모르겠다

# loss :  0.0681016817688942
# acc : 0.9777777791023254
#텐서 난수웨이트값 적용

# loss :  0.20700840651988983
# acc : 0.9777777791023254
# acc 0.9777777777777777

#민맥스
# loss :  0.11016491055488586
# acc : 0.9777777791023254
# acc 0.9777777777777777

#스탠
# loss :  0.0687570571899414
# acc : 1.0
# acc 1.0

#인상적인 스탠의 능력

# dropout
# loss :  0.03909888491034508
# acc : 1.0
# acc 1.0

# lstm
# loss :  1.0712865591049194
# acc : 0.35555556416511536
# acc 0.35555555555555557
