import tensorflow as tf
tf.compat.v1.set_random_seed(324)

#1. 데이터

x1_data = [23.,12.,1235.,213.,123.]
x2_data = [31.,34.,38.,0.,123.]
x3_data = [23.,12.,438.,38.,38.]
y_data = [23.,12.,1235.,213.,123.]

x1 = tf.placeholder(tf.float32,shape=[None])
x2 = tf.placeholder(tf.float32,shape=[None])
x3 = tf.placeholder(tf.float32,shape=[None])
y = tf.placeholder(tf.float32, shape=[None])

w1 = tf.compat.v1.Variable(tf.random_normal([1]),tf.float32)
w2 = tf.compat.v1.Variable(tf.random_normal([1]),tf.float32)
w3 = tf.compat.v1.Variable(tf.random_normal([1]),tf.float32)
b = tf.compat.v1.Variable(tf.random_normal([1]),tf.float32)

hypothesis = x1*w1 +x2*w2 +x3*w3 + b

#3-1 컴파일
loss = tf.reduce_mean(tf.square(hypothesis-y_data))
opti = tf.train.GradientDescentOptimizer(learning_rate=0.000000000000000001).minimize(loss)

loss_val_list = []
W_val_list = []

with tf.compat.v1.Session() as sess:
    epochs = 100
    sess.run(tf.global_variables_initializer())
    for step in range(epochs):
        _,loss_val,w1_val,w2_val,w3_val,b_val,h_val = sess.run([opti,loss,w1,w2,w3,b,hypothesis],feed_dict = {x1:x1_data,x2:x2_data,x3:x3_data,y:y_data})
        if step % 10 == 0:
            print(step,'\t',loss_val,'\t',w1,'\t',w2,'\t',w3,'\t',b_val,'\n',h_val)

pred = x1*w1+x2*x2+x3*w3+b

x_test_data = [14.,13.,2342.,324.,2341]
x_test = tf.placeholder(tf.float32,shape=[None])
from sklearn.metrics import r2_score

pred = h_val * x_test

print(r2_score(y_data,h_val))