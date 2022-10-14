from this import d
import tensorflow as tf
tf.compat.v1.set_random_seed(3242)

x_train = [1]
y_train = [1]
x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)
w = tf.Variable(10,dtype=tf.float32)
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(w))
hypothesis = x*w
loss = tf.reduce_mean(tf.square(hypothesis-y))

lr = 0.2
gradient = tf.reduce_mean((w*x-y)*x)
decent = w- lr *gradient
update = w.assign(decent)
#계산하고 러닝레이트 바꿔서 계산
w_history = []
loss_history = []


for step in range(21):

    _,loss_v,w_v = sess.run([update,loss,w],feed_dict={x:x_train,y:y_train})
    
    print(step,'\t',_,'\t',loss_v,'\t',w_v)

    w_history.append(w_v)
    loss_history.append(loss_v)
print(sess.run(w))
sess.close()
print('========================')
print(w_history)
print('========================')
print(loss_history)
