import tensorflow as tf
sess = tf.Session()
# tf.executing_eagerly()

x = tf.Variable([2], dtype=tf.float32)
y = tf.Variable([3], dtype=tf.float32)


# sess.run(x+y)
# tensorflow.python.framework.errors_impl.FailedPreconditionError: Attempting to use uninitialized value Variable

init = tf.compat.v1.global_variables_initializer()
sess.run(init)
print(sess.run(x+y))
print(tf.__version__)