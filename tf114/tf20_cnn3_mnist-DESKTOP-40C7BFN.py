import tensorflow as tf
import keras
import numpy as np
tf.compat.v1.disable_eager_execution()

tf.compat.v1.set_random_seed(123)

#1.data
from keras.datasets import mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()
from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_test.shape)
x_train = x_train.reshape(-1,28,28,1).astype('float32')/255.
x_test = x_test.reshape(-1,28,28,1).astype('float32')/255.

#2.모델
# 128
# 64
# 32
# flatten
#100
#10

x = tf.compat.v1.placeholder(tf.float32,[None,28,28,1]) #input_shape
y = tf.compat.v1.placeholder(tf.float32,[None,10])

#layer1
w1 = tf.compat.v1.get_variable('w1',shape=[2,2,1,128])
L1 = tf.nn.conv2d(x,w1,strides=[1,1,1,1],padding='SAME')
#model.add(Conv2d(64,kerner_size=(2,2),input_shape=(28,28,1))) stride 기본값 1, 1칸씩
# print(w1) #<tf.Variable 'w1:0' shape=(2, 2, 1, 64) dtype=float32_ref>
print(L1) #shape=(?, 28, 28, 128)
L1 = tf.nn.relu(L1)
#activation='relu'
L1_maxpool = tf.nn.max_pool2d(L1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

print(L1_maxpool) #(?, 14, 14, 128)
#layer2
w2 = tf.compat.v1.get_variable('w2',shape=[3,3,128,64])
L2 = tf.nn.conv2d(L1_maxpool,w2,strides=[1,1,1,1],padding='VALID')
L2 = tf.nn.selu(L2)
L2_maxpool = tf.nn.max_pool2d(L2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
print(L2_maxpool)

w3 = tf.compat.v1.get_variable('w3',shape=[3,3,64,32])
L3 = tf.nn.conv2d(L2_maxpool,w3,strides=[1,1,1,1],padding='VALID')
L3 = tf.nn.elu(L3)
print(L3)

L_Flatten = tf.reshape(L3,[-1,4*4*32])
print(L_Flatten)
# import tensorflow.compat.v1 as tf 
# tf.disable_v2_behavior() 
w4 = tf.compat.v1.get_variable('w4',shape=[4*4*32,100], initializer=tf.compat.v1.keras.initializers.glorot_noraml())
b4 = tf.Variable(tf.random.normal([100]),name='b4')
L4 = tf.nn.selu(tf.matmul(L_Flatten,w4)+b4)
rate = tf.compat.v1.placeholder(tf.float32)
L4 = tf.nn.dropout(L4,rate=0.3)

w5 = tf.compat.v1.get_variable('w5',shape=[100,10], initializer=tf.compat.v1.keras.initializers.glorot_noraml())
b5 = tf.Variable(tf.random.normal([10]),name='b5')
L5 = tf.matmul(L4,w5)+b5
hypothesis = tf.nn.softmax(L5)

print(hypothesis)

loss = tf.reduce_mean(-tf.reduce_mean(y*tf.compat.v1.log(hypothesis),axis=1))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis,labels=y))
train = tf.compat.v1.train.AdadeltaOptimizer(learning_rate=0.01).minimize(loss,var_list=[w5,b5])


epochs = 20
batch_size = 100
total_batch = int(len(x_train)/batch_size)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(epochs): #30번
    avg_loss = 0
    for i in range(total_batch): #600번
        start = i*batch_size # 0
        end = start+batch_size #100
        batch_x, batch_y = x_train[start:end], y_train[start:end]
        feed_dict = {x:batch_x,y:batch_y,rate:0.3}
        batch_loss, _ =sess.run([loss,train],feed_dict=feed_dict)
        avg_loss += batch_loss /total_batch
        print(batch_loss.shape)
    print('epoch:','%04d'%(epoch+1),'loss{:.9f}'.format(avg_loss))
print('end')

pred = tf.equal(tf.arg_max(hypothesis,1),tf.argmax(y,1))
acc = tf.reduce_mean(tf.cast(pred,tf.float32))
print('acc',sess.run(acc,feed_dict={x:x_test,y:y_test, rate:0.0}))