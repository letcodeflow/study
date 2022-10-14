import tensorflow as tf
import numpy as np
from sklearn.datasets  import load_wine

#1.데이터
datasets = load_wine()
x = datasets.data
y = datasets.target.reshape(-1,1)
print(x.shape, y.shape)
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
encoder.fit(y)
y = encoder.transform(y).toarray()
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, shuffle=True, random_state=137)

from tensorflow.python.keras.layers import Dense, Dropout, LSTM
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping
ES = EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True)
model = Sequential()
model.add(LSTM(5, input_shape=(1,13)))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))

from sklearn.metrics import accuracy_score
x = tf.compat.v1.placeholder(tf.float32,shape=[None,13])
y = tf.compat.v1.placeholder(tf.float32,shape=[None,3])

w = tf.Variable(tf.random.normal([13,20]),tf.float32)
b = tf.Variable(tf.random.normal([20]),tf.float32)
hidden = tf.matmul(x,w)+b

w2 = tf.Variable(tf.random.normal([20,21]),tf.float32)
b2 = tf.Variable(tf.random.normal([21]),tf.float32)
hidden2 = tf.matmul(hidden,w2)+b2

w3 = tf.Variable(tf.random.normal([21,23]),tf.float32)
b3 = tf.Variable(tf.random.normal([23]),tf.float32)
hidden3 = tf.matmul(hidden2,w3)+b3

w4 = tf.Variable(tf.random.normal([23,3]),tf.float32)
b4 = tf.Variable(tf.random.normal([3]),tf.float32)
hypothesis = tf.nn.softmax(tf.matmul(hidden3,w4)+b4)
loss = tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20):
        _ = sess.run(train, feed_dict={x:x_train,y:y_train})
        if i % 2==0:
            print(i)
    pred = sess.run(hypothesis,feed_dict={x:x_test})
    pred = np.argmax(pred,axis=1)
    y_test = np.argmax(y_test,axis=1)
    print(accuracy_score(y_test,pred))
#3.컴파일 훈련
# model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# hist = model.fit(x_train, y_train, epochs=1, verbose=1, 
#           validation_split=0.2,
#           callbacks=ES)

# #4.평가 예측
# loss = model.evaluate(x_test, y_test)

# y_predict = model.predict(x_test)
# # y_predict = model.predict(x_test[:5])
# print(y_test)
# print(y_predict)

# y_predict = np.argmax(y_predict, axis= 1)
# y_test = np.argmax(y_test, axis= 1)
# print(y_test)
# print(y_predict)

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

#민맥스
# loss :  0.24404345452785492
# acc : 0.9629629850387573
# acc 0.9629629629629629

#스탠
# loss :  0.12898489832878113
# acc : 0.9629629850387573
# acc 0.9629629629629629

#스케일미적용
# loss :  0.7572556138038635
# acc : 0.6481481194496155
# acc 0.6481481481481481

# dropout
# loss :  1.0873541831970215
# acc : 0.29629629850387573
# acc 0.2962962962962963

# lstm
# loss :  1.091688632965088
# acc : 0.40740740299224854
# acc 0.4074074074074074