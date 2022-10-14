import tensorflow as tf
tf.compat.v1.set_random_seed(3242)

x = [1,2,3]
y = [1,2,3]
w = tf.compat.v1.placeholder(tf.float32)
hypothesis = x*w
loss = tf.reduce_mean(tf.square(hypothesis-y))

w_history = []
loss_history = []

with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(-30,50):
        curr_w = i
        curr_loss = sess.run(loss, feed_dict={w:curr_w})

        w_history.append(curr_w)
        loss_history.append(curr_loss)

print('========================')
print(w_history)
print('========================')
print(loss_history)
import matplotlib.pyplot as plt
plt.plot(w_history,loss_history)
plt.xlabel('weight')
plt.ylabel('loss')
plt.show()