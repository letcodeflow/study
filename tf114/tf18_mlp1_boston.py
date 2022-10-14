from sklearn import metrics
from sklearn.datasets import load_boston
import tensorflow as tf
from sklearn.metrics import r2_score
data_sets = load_boston()

x = data_sets.data
y = data_sets.target

print(x.shape, y.shape)
# x = x.reshape(-1,13)
y= y.reshape(-1,1)
#sklearn dataset만 먹는 명령어
# print(data_sets.DESCR)

# Number of Instances: 506

#     :Number of Attributes: 13 

#x,y 는 이미 설정돼있음 모델 만들면 됨 train test 셋 나눠줘야함

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=False)

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D,LSTM
# x_train=x_train.reshape(-1,13,1)
# x_test=x_test.reshape(-1,13,1)

print(x.shape)
model = Sequential()
model.add(LSTM(5, input_shape=(13,1)))
model.add(Dense(5, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(5, activation='sigmoid'))
model.add(Dense(5))
model.add(Dense(1))
x = tf.compat.v1.placeholder(tf.float32,shape=[None,13])
y = tf.compat.v1.placeholder(tf.float32,shape=[None,1])

w1 = tf.Variable(tf.random.normal([13,20]),tf.float32)
b1 = tf.Variable(tf.random.normal([20]),tf.float32)
hidden1 = tf.matmul(x,w1)+b1

w2 = tf.Variable(tf.random.normal([20,30]),tf.float32)
b2 = tf.Variable(tf.random.normal([30]),tf.float32)
hidden2 = tf.matmul(hidden1,w2)+b2

w3 = tf.Variable(tf.random.normal([30,40]),tf.float32)
b3 = tf.Variable(tf.random.normal([40]),tf.float32)
hidden3 = tf.matmul(hidden2,w3)+b3

w4 = tf.Variable(tf.random.normal([40,50]),tf.float32)
b4 = tf.Variable(tf.random.normal([50]),tf.float32)
hidden4 = tf.matmul(hidden3,w4)+b4

w5 = tf.Variable(tf.random.normal([50,60]),tf.float32)
b5 = tf.Variable(tf.random.normal([60]),tf.float32)
hidden5 = tf.matmul(hidden4,w5)+b5

w6 = tf.Variable(tf.random.normal([60,1]),tf.float32)
b6 = tf.Variable(tf.random.normal([1]),tf.float32)
hypothesis = tf.matmul(hidden5,w6)+b6
loss = tf.reduce_mean(tf.square(hypothesis-y))
train = tf.train.GradientDescentOptimizer(learning_rate=0.00000000000000000000000000001).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20):
        _ = sess.run(train,feed_dict={x:x_train,y:y_train})
        if i % 20 == 0:
            print(i)
    pred = sess.run(hypothesis,feed_dict={x:x_test})
    print(r2_score(y_test,pred))    
#컴파일 훈련
# model.compile(loss = 'mse', optimizer = 'adam', metrics=['accuracy', 'mae'])
# #컴파일 전후 얼리스타핑 미니멈 혹은 맥시멈값을 patience 지켜보고 있다가 정지시키는 함수

# from tensorflow.python.keras.callbacks import EarlyStopping
# ES = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=2, restore_best_weights=True)

# #훈련을 인스턴스 하나로 줄여주기
# h = model.fit(x_train, y_train, epochs=100, batch_size=10,
#           validation_split=0.1, 
#           callbacks=[ES], 
#           verbose=3)

# #훈련돼ㅆㅇ니 평가 예측

# loss = model.evaluate(x_test, y_test)
# print('loss', loss)

# print(h) #훈련 h가 어느 메모리에 저장돼있는가
# print('=======================================')
# print(h.history) #훈련 h의 loss율 그리고 추가돼있다면 validation loss 율\

# y_predict = model.predict(x_test)

# #y예상값이 나왔으므로 실제 y값과 비교해본다
# from sklearn.metrics import r2_score
# r2= r2_score(y_test, y_predict)
# print('r2', r2)
# print('loss', loss)

#그리기


#vali 미적용
# r2 -0.5474061321360475
# loss [41.60723114013672, 0.0, 5.335869312286377]

# #적용
# r2 0.05849269602291585
# loss [25.31559944152832, 0.0, 4.08815336227417]

# 사이즈가 작은데도 상당히 큰 효과가 있었다. 리니어모델이라서 그런가?

# 드롭아웃
# r2 -0.009316729116613631
# loss [27.138879776000977, 0.0, 4.281790256500244]

# r2 -0.0663786709369325
# loss [28.67318344116211, 0.0, 4.445716857910156]

# lstm
# r2 -0.07838065403036132
# loss [28.99589729309082, 0.0, 4.46611213684082]

