import numpy as np
import tensorflow as tf
tf.compat.v1.set_random_seed(124)
from sklearn.metrics import accuracy_score
#1.데이터
x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [[0],[1],[1],[0]]

#2.모델
#input layer
x = tf.compat.v1.placeholder(tf.float32,shape=[None,2])
y = tf.compat.v1.placeholder(tf.float32,shape=[None,1])
#hidden layer
w1 = tf.compat.v1.Variable(tf.random_normal([2,20]),tf.float32)
b1 = tf.compat.v1.Variable(tf.random_normal([20]),tf.float32)
hidden2 = tf.matmul(x,w1)+b1
# hypothesis = (x@w1+b1)*w2+b2
#output layer
w2 = tf.compat.v1.Variable(tf.random_normal([20,30]),tf.float32)
b2 = tf.compat.v1.Variable(tf.random_normal([30]),tf.float32)

hidden3 = tf.matmul(hidden2,w2)+b2

w3 = tf.compat.v1.Variable(tf.random_normal([30,300]),tf.float32)
b3 = tf.compat.v1.Variable(tf.random_normal([300]),tf.float32)
hidden4 = tf.matmul(hidden3,w3)+b3

w4 = tf.compat.v1.Variable(tf.random_normal([300,8000]),tf.float32)
b4 = tf.compat.v1.Variable(tf.random_normal([8000]),tf.float32)
hidden5 = tf.matmul(hidden4,w4)+b4

w5 = tf.compat.v1.Variable(tf.random_normal([8000,10000]),tf.float32)
b5 = tf.compat.v1.Variable(tf.random_normal([10000]),tf.float32)
hidden6 = tf.matmul(hidden5,w5)+b5

w6 = tf.compat.v1.Variable(tf.random_normal([10000,1]),tf.float32)
b6 = tf.compat.v1.Variable(tf.random_normal([1]),tf.float32)
hypothesis = tf.sigmoid(tf.matmul(hidden6,w6)+b6)

loss = tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.000000005).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        _,h_val = sess.run([train,hypothesis],feed_dict={x:x_data,y:y_data})
        if i % 2 ==0:
            print(i, h_val)
    pred = sess.run(hypothesis,feed_dict={x:x_data})
    print(pred)
    print(accuracy_score(y_data,np.round(pred)))


