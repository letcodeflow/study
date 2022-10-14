import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
x = tf.compat.v1.placeholder(tf.float32,shape=[None,2])
y = tf.compat.v1.placeholder(tf.float32,shape=[None,30])

w1 = tf.compat.v1.Variable(tf.random.normal([2,30]),name='weights1')
b1 = tf.compat.v1.Variable(tf.random.normal([30]),name='bias1')

hidden1 = tf.compat.v1.sigmoid(tf.matmul(x*w1)+b1)
#model.add(Dense(30,input_shape=(2,),activation='sigmoid'))

dropout = tf.compat.v1.nn.dropout(hidden1,keep_prob=0.7)
print(hidden1) #
print(dropout)