
# y = wx+b

import tensorflow as tf
tf.set_random_seed(14)
print(tf.__version__)
#1.데이터

x_train_data= [1,2,3]
y_train_data= [3,5,7]

x_train = tf.placeholder(tf.float32,shape=[None])
y_train = tf.placeholder(tf.float32,shape=[None])
#웨이트값 난수로
W = tf.Variable(tf.random_normal([1]),tf.float32)
b = tf.Variable(tf.random_normal([1]),dtype=tf.float32)
model = x_train*W+b

loss = tf.reduce_mean(tf.square(model-y_train))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.176)

train = optimizer.minimize(loss)

epochs = 101
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(epochs):
        _,loss_val,W_val,b_val = sess.run([train, loss,W,b],feed_dict={x_train:x_train_data,y_train :y_train_data})
        if step % 10 == 0:
            print(step,loss_val,W_val,b_val)

######################predict
    x_test_data = [6,7,8]
    x_test = tf.placeholder(tf.float32,shape=[None])

    pred = x_test * W_val + b_val
    # sess = tf.Session()
    # sess.run(pre,feed_dict=x_data,y_test)

    # pred = model.predict(x_test)
    print('[6,7,8] 예측', sess.run(pred,feed_dict={x_test:x_test_data}))
