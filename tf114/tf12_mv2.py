import tensorflow as tf
tf.compat.v1.set_random_seed(324)

x_data = [[73,51,134],
        [23,243,51],
        [1234,234,67],
        [132,234,2354],
        [23,54,34] ]
y_data = [[152],[342],[34],[435],[4325]]

x = tf.compat.v1.placeholder(tf.float32,shape=[None,3])
y = tf.compat.v1.placeholder(tf.float32,shape=[None,1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([3,1]),name='w')
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]),name='b')

hypothesis = tf.compat.v1.matmul(x,w)+b

opti_train_loss = tf.train.GradientDescentOptimizer(learning_rate=0.0000001).minimize(tf.reduce_mean(tf.square(hypothesis-y_data)))

with tf.compat.v1.Session() as sess:
    epochs = 100
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        hy_val,loss_val,_ = sess.run([hypothesis,tf.reduce_mean(tf.square(hypothesis-y_data)),opti_train_loss], feed_dict = {x:x_data,y:y_data})
        if epoch % 10 == 0:
            print(epoch,'\n',hy_val,'\n',loss_val)

from sklearn.metrics import r2_score
print(r2_score(y_data, hy_val))
