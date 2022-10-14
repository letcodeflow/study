import tensorflow as tf
import numpy as np
print(tf.__version__)
tf.executing_eagerly()


node1 = tf.constant(3.0)
node2 = tf.constant(4.0)
node3 = tf.add(node1,node2)
sess = tf.compat.v1.Session()

a = tf.compat.v1.placeholder(tf.float32)
b = tf.compat.v1.placeholder(tf.float32)


add_node = a+b

print(sess.run(add_node, feed_dict={a:[3,3], b:[4.5,6]}))

triple = add_node*3
print(triple)
print(sess.run(triple, feed_dict={a:[3,3], b:[4.5,6]}))
