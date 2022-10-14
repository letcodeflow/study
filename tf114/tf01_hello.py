import tensorflow as tf
print(tf.__version__)

hell  = tf.constant('hello world')
print(hell)

# sess = tf.Session()
sess = tf.compat.v1.Session()
print(sess.run(hell))