import numpy as np
import tensorflow as tf
tf.compat.v1.set_random_seed(124)
from sklearn.metrics import accuracy_score
#1.데이터
x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [[0],[1],[1],[0]]

#2.모델
x = tf.compat.v1.placeholder(tf.float32,shape=[None,2])
w = tf.compat.v1.Variable(tf.random_normal([2,1]),tf.float32)
b = tf.compat.v1.Variable(tf.random_normal([1]),tf.float32)
y = tf.compat.v1.placeholder(tf.float32,shape=[None,1])

hypothesis = tf.sigmoid(tf.matmul(x,w)+b)
loss = tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10):
        _,h_val = sess.run([train,hypothesis],feed_dict={x:x_data,y:y_data})
        if i % 2 ==0:
            print(i, h_val)
    pred = sess.run(tf.cast(h_val>=0.5, dtype=tf.float32))
    print(pred)
    print(accuracy_score(y_data,pred))


