from sklearn.datasets import fetch_california_housing
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import datetime
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
tf.set_random_seed(1213)
# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target.reshape(-1,1)

x_train, x_test, y_train, y_test =  train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)
print(x_train.shape, x_test.shape)

# Scaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# x_train = x_train.reshape(16512, 4, 2) # 데이터의 갯수자체는 성능과 큰 상관이 없을 수 있다
# x_test = x_test.reshape(4128, 4, 2)

# 2. 모델구성
model = Sequential()
model.add(LSTM(10, input_shape=(4,2)))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
# model.summary()

print(y_train.shape)
import tensorflow as tf
x = tf.compat.v1.placeholder(tf.float32,shape=[None,8])
y = tf.compat.v1.placeholder(tf.float32,shape=[None,1])

w1 = tf.Variable(tf.random.normal([8,20]),tf.float32)
b1 = tf.Variable(tf.random.normal([20]),tf.float32)
hidden1 = tf.matmul(x,w1)+b1

w2 = tf.Variable(tf.random.normal([20,40]),tf.float32)
b2 = tf.Variable(tf.random.normal([40]),tf.float32)
hidden2 = tf.matmul(hidden1,w2)+b2

w3 = tf.Variable(tf.random.normal([40,20]),tf.float32)
b3 = tf.Variable(tf.random.normal([20]),tf.float32)
hidden3 = tf.matmul(hidden2,w3)+b3

w4 = tf.Variable(tf.random.normal([20,1]),tf.float32)
b4 = tf.Variable(tf.random.normal([1]),tf.float32)
hypothesis = tf.matmul(hidden3,w4)+b4

loss = tf.reduce_mean(tf.square(hypothesis-y))
train = tf.train.GradientDescentOptimizer(learning_rate=0.0000000000000000000001).minimize(loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20):
        _ = sess.run(train,feed_dict={x:x_train,y:y_train})
        if i % 20 ==0:
            print(i)
    pred = sess.run(hypothesis,feed_dict={x:x_test})
    print(r2_score(y_test,pred))

# 3. 컴파일, 훈련
# model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])
# earlyStopping = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1, restore_best_weights=True)
# hist = model.fit(x_train, y_train, epochs=200, batch_size=50,
#                 callbacks=[earlyStopping],
#                 validation_split=0.25)

# # 4. 평가, 예측
# loss = model.evaluate(x_test, y_test)
# print("loss : ", loss)

# y_predict = model.predict(x_test)
# r2 = r2_score(y_test, y_predict)
# print('r2스코어 : ', r2)

# loss :  [0.5216111540794373, 0.546445369720459]
# r2스코어 :  0.625834467060862

# lstm
# loss :  [0.3997993469238281, 0.44844144582748413]
# r2스코어 :  0.7132133703245158