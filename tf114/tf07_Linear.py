# y = wx+b

import tensorflow as tf
tf.set_random_seed(14)

#1.데이터
x = [1,2,3]
y = [1,2,3]

W = tf.Variable(1,dtype=tf.float32)
b = tf.Variable(1,dtype=tf.float32)

#2.모델
hypothesis = x*W + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)\

train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(train)
    if step %20 == 0:
        print(step, sess.run(loss),sess.run(W),sess.run(b))

sess.close()