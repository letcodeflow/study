#1.데이터
from sklearn.datasets import fetch_california_housing
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target.reshape(-1,1)

print(x.shape)
print(y.shape)



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=66)
#스케일링팥
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_train))
print(np.max(x_train))
print(np.min(x_test))
print(np.max(x_test))
#minmax
# 0.0
# 1.0
# -0.0050864699898269805
# 2.0745532963647566
#stan
# -2.369187755757429
# 109.92119256376915
# -2.3942132659109805
# 228.48436764973988

#2.모델
import tensorflow as tf
x = tf.compat.v1.placeholder(tf.float32,shape=[None,8])
y = tf.compat.v1.placeholder(tf.float32,shape=[None,1])

w = tf.Variable(tf.random.normal([8,20]),tf.float32)
b = tf.Variable(tf.random.normal([20]),tf.float32)
hidden = tf.matmul(x,w)+b

w2 = tf.Variable(tf.random.normal([20,30]),tf.float32)
b2 = tf.Variable(tf.random.normal([30]),tf.float32)
hidden2 = tf.matmul(hidden,w2)+b2

w3 = tf.Variable(tf.random.normal([30,10]),tf.float32)
b3 = tf.Variable(tf.random.normal([10]),tf.float32)
hidden3 = tf.matmul(hidden2,w3)+b3

w4 = tf.Variable(tf.random.normal([10,1]),tf.float32)
b4 = tf.Variable(tf.random.normal([1]),tf.float32)
hypothesis = tf.matmul(hidden3,w4)+b4
loss = tf.reduce_mean(tf.square(hypothesis-y))
train = tf.train.GradientDescentOptimizer(learning_rate=0.0000000000000000000001).minimize(loss)
from sklearn.metrics import r2_score
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20):
        _ = sess.run(train,feed_dict={x:x_train,y:y_train})
        if i % 2==0:
            print(i)
    pred = sess.run(hypothesis,feed_dict={x:x_test})
    print(r2_score(y_test,pred))






from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, LSTM

model=Sequential()
model.add(LSTM(20, input_shape=(2,4)))
model.add(Dense(80))
model.add(Dense(20))
model.add(Dropout(0.3))
model.add(Dense(80))
model.add(Dense(20))
model.add(Dropout(0.3))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dropout(0.3))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dropout(0.2))
model.add(Dense(20))
model.add(Dense(1))

#3. 컴파일 훈련
# model.compile(loss = 'mse', optimizer='adam')
# model.fit(x_train, y_train, epochs=250, batch_size=200, validation_split=0.01)

# #4. 평가, 예측
# loss = model.evaluate(x_test, y_test)
# print('loss', loss)

# y_predict = model.predict(x_test)

# from sklearn.metrics import r2_score
# r2=r2_score(y_test, y_predict)
# print('r2', r2)

#validation 이전

# loss 0.6581024527549744
# 194/194 [==============================] - 0s 664us/step
# r2 0.5203931464750856

# loss 0.6531566381454468
# 194/194 [==============================] - 0s 674us/step
# r2 0.5239972327558571

# loss 0.650373637676239
# 194/194 [==============================] - 0s 639us/step
# r2 0.5260254847020475

#적용후
# loss 0.6490857005119324
# 194/194 [==============================] - 0s 615us/step
# r2 0.5269640664307653

# loss 0.6621416807174683
# 194/194 [==============================] - 0s 632us/step
# r2 0.5174494649013497

# loss 0.7691570520401001
# 194/194 [==============================] - 0s 654us/step
# r2 0.4394595104425515

#minmax
# loss 0.5488771796226501
# r2 0.5999933877663844

#stan
# loss 0.5570531487464905
# r2 0.594034815118113

#조금 높아졌음

# Dropout
# loss 0.5472083687782288
# r2 0.60120947610524

# lstm
# loss 0.3066251277923584
# r2 0.7765400196338007