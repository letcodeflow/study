import tensorflow as tf
import keras
import numpy as np

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
x = tf.compat.v1.placeholder(tf.float32,[None,28,28,1]) #input_shape
y = tf.compat.v1.placeholder(tf.float32,[None,10])

w1 = tf.compat.v1.get_variable('w1',shape=[2,2,1,64])

L1 = tf.nn.conv2d(x,w1,strides=[1,1,1,1],padding='VALID')
#model.add(Conv2d(64,kerner_size=(2,2),input_shape=(28,28,1))) stride 기본값 1, 1칸씩
print(w1) #<tf.Variable 'w1:0' shape=(2, 2, 1, 64) dtype=float32_ref>
print(L1)



