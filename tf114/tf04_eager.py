import tensorflow as tf
print(tf.__version__)
print(tf.executing_eagerly())

# tf.compat.v1.disable_eager_execution()
print(tf.compat.v1.disable_eager_execution())

# False
# None 
#2.0버전에서는 Ture False값이 나옴 = 즉 2.0은 즉시실행상태라는 뜻

hello = tf.constant('hello')
sess = tf.compat.v1.Session()
print(sess.run(hello))