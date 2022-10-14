import tensorflow as tf
tf.compat.v1.set_random_seed(324)

#1.데이터
x_data = [[1,2],[2,3],[3,4],[4,5],[5,6],[6,7]] #6,2
y_data = [[0],[0],[0],[1],[1],[1]] #6,1

print(len(x_data))

x = tf.compat.v1.placeholder(tf.float32,shape=[None,2])
y = tf.compat.v1.placeholder(tf.float32,shape=[None,1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([2,1]))
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]))

hypothesis =tf.compat.v1.sigmoid(tf.compat.v1.matmul(x,w)+b)

loss = tf.reduce_mean(tf.square(hypothesis-y_data))
loss = tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))
opti = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss)

with tf.compat.v1.Session() as sess:
    epochs = 101
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        hy_val,loss_val,_ = sess.run([hypothesis,loss,opti],feed_dict={x:x_data,y:y_data})
        if epoch % 10 == 0:
            print(epoch, hy_val,'\n',loss_val,'\n',_)
sess = tf.InteractiveSession()
pred = sess.run(tf.cast(hy_val>0.5,dtype=tf.float32))
from sklearn.metrics import accuracy_score
print(accuracy_score(y_data,pred))