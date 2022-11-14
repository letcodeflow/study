import numpy as np
import tensorflow as tf
import torch.nn as nn
# ini = tf.initializers.Ones()
# kernels = tf.Variable(ini([1,2,2,1]),trainable=False)
x = tf.ones((1,1,3,3))
print(x)
conv2d = tf.keras.layers.Conv2D(3,1,kernel_initializer=tf.initializers.ones())(x)
print('conv2d',conv2d)
# (1, 2, 2, 1)
conv2dTranspose = tf.keras.layers.Conv2DTranspose(3,1,kernel_initializer=tf.initializers.ones())(x)
upsample = tf.keras.UpSampling2d(2)(x)
print('conv2dTranspose',conv2dTranspose)
# (1, 4, 4, 1)
# kernel = tf.constant_initializer(1.)
# x = tf.ones((1,3,3,1))
# conv = tf.keras.layers.Conv2D(1,2, kernel_initializer=kernel)
# y = tf.ones((1,2,2,1))

# de_conv = tf.keras.layers.Conv2DTranspose(1,2, kernel_initializer=kernel)

# conv_output = conv(x)
# print("Convolution\n---------")
# print("input  shape:",x.shape)
# print("output shape:",conv_output.shape)
# print("input  tensor:",np.squeeze(x.numpy()).tolist())
# print("output tensor:",np.around(np.squeeze(conv_output.numpy())).tolist())
# '''
# Convolution
# ---------
# input  shape: (1, 3, 3, 1)
# output shape: (1, 2, 2, 1)
# input  tensor: [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
# output tensor: [[4.0, 4.0], [4.0, 4.0]]
# '''
# de_conv_output = de_conv(y)
# print("De-Convolution\n------------")
# print("input  shape:",y.shape)
# print("output shape:",de_conv_output.shape)
# print("input  tensor:",np.squeeze(y.numpy()).tolist())
# print("output tensor:",np.around(np.squeeze(de_conv_output.numpy())).tolist())
# '''
# De-Convolution
# ------------
# input  shape: (1, 2, 2, 1)
# output shape: (1, 3, 3, 1)
# input  tensor: [[1.0, 1.0], [1.0, 1.0]]
# output tensor: [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]
# '''