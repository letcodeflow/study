
# y = wx+b

import tensorflow as tf
# tf.set_random_seed(14)
print(tf.__version__)
#1.데이터
x = [1,2,3,4,5]
y = [1,2,3,4,5]


#웨이트값 난수로
W = tf.Variable(tf.random_normal([1]),dtype=tf.float32)
b = tf.Variable(tf.random_normal([1]),dtype=tf.float32)
sess = tf.Session()
model = x*W+b

loss = tf.reduce_mean(tf.square(model-y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

epochs = 2001
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(epochs):
        sess.run(train)
        if step % 200 == 0:
            print(step,sess.run(loss),sess.run(W),sess.run(b))




