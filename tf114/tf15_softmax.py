import numpy as np
import tensorflow as tf
tf.set_random_seed(32)

x_data = [[1,2,3,54],
[2,234,2,1],
[2,3,5,2],
[234,2,3,32],
[2,234,2,1],
[2,3,5,2],
[2,3,5,2],
[234,2,3,32],
]
#8,4
y_data = [[0,0,1],
[0,0,1],
[0,0,1],
[0,1,1],
[0,1,1],
[0,1,1],
[1,0,0],
[1,0,0],]
# 8,3
#2.모델구성

# 원핫상태
#1레이어
x = tf.compat.v1.placeholder(tf.float32,shape=[None,4])
w = tf.Variable(tf.random_normal([4,3]))
b = tf.Variable(tf.random_normal([1,3]))
y = tf.compat.v1.placeholder(tf.float32,shape=[None,3])


hypothesis = tf.nn.softmax(tf.matmul(x,w)+b)

#3-1.컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis),axis=1))
train = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

for i in range(201):
    h_val,loss_val,_ = sess.run([hypothesis,loss,train],feed_dict={x:x_data,y:y_data})
    if i % 20==0:
        print(i,h_val,loss_val)
    
    pred = sess.run(hypothesis,feed_dict={x:x_data})
    print(pred, sess.run(tf.argmax(pred,1)))
    from sklearn.metrics import accuracy_score
    print(accuracy_score(y_data,sess.run()))
