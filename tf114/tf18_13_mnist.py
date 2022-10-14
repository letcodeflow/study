#dnn

import tensorflow as tf
from keras.datasets import mnist
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
import numpy as np

# model = Sequential()
# model.add(Dense(64, input_shape=(28*28,)))
# model.add(Dense(64, input_shape=(784,)))
# #x 쉐잎이 784 혹은 28*28이 되게한다a
# model.add(Dense(10,activation='softmax'))

# model.compile(loss = 'categorical_crossentropy', optimizer='adam')
# model.fit(x_train, y_train, epochs=10, batch_size=20000)
(x_train,y_train),(x_test,y_test) = mnist.load_data()
from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(-1,x_train.shape[1]*x_train.shape[2]).astype/255.
x_test = x_test.reshape(-1,x_test.shape[1]*x_test.shape[2]).astype/255.

print(x_train.shape,y_train.shape)
x = tf.compat.v1.placeholder(tf.float32,shape=[None,x_train.shape[1]*x_train.shape[2]])
y = tf.compat.v1.placeholder(tf.float32,shape=[None,10])

w1 = tf.compat.v1.get_variable('w1',shape=[x_train.shape[1]*x_train.shape[2],64])
b1 = tf.compat.v1.get_variable('w1',shape=[64])
hidden1 = tf.compat.v1.nn.relu(tf.compat.v1.matmul(x, w1) + b1)

w2 = tf.compat.v1.Variable(tf.compat.v1.zeros([64,32]), name='weight')
b2 = tf.compat.v1.Variable(tf.compat.v1.zeros([32]), name='bias')
hidden2 = tf.compat.v1.matmul(hidden1, w2) + b2

w3 = tf.compat.v1.Variable(tf.compat.v1.zeros([32,10]), name='weight')
b3 = tf.compat.v1.Variable(tf.compat.v1.zeros([10]), name='bias')
hypothesis = tf.compat.v1.nn.softmax(tf.compat.v1.matmul(hidden2, w3) + b3)

# hidden1 = tf.nn.softmax(tf.matmul(x,w1)+b1)
# dropout = tf.compat.v1.nn.dropout(hidden1,rate=0.5)
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis=1))

train = tf.train.AdadeltaOptimizer(learning_rate=1e-5).minimize(loss)


sess = tf.Session()
output = sess.run(hidden1,feed_dict={x:x_train,y:y_train})

epochs = 1001
for step in range(epochs):
    _, hy_val, cost_val, b_val = sess.run([train,hypothesis,loss,b3], feed_dict={x:x_train, y:y_train})
    if step%20 == 0:
        print(step, cost_val, hy_val)
        
print('최종: ', cost_val, hy_val)

_, y_pred = sess.run([train,hypothesis], feed_dict={x:x_test, y:y_test})

acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
print('acc: ', acc)


