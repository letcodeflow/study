import tensorflow as tf
.
node1 = tf.constant(3.0,tf.float32)
node2 = tf.constant(4.0)

node3 = tf.subtract(node1,node2)
node4 = tf.multiply(node1,node2)
# node5 = tf.matmul(node1,node2)
node6 = tf.divide(node1,node2)
# node7 = tf.mod(node1,node2)

node8 = node1 % node2
node9 = node1 // node2


print(tf.Session().run(node3))
print(tf.Session().run(node4))
# print(tf.Session()(node5))
print(tf.Session().run(node6))
# print(tf.Session().run(node7))
print(tf.Session().run(node8))
print(tf.Session().run(node9))