
import numpy as np
import tensorflow as tf
tf.debugging.set_log_device_placement(True)
#1.데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,3,5,4,7,6,7,11,9,7])
# tf.compat.v1.disable_eager_execution()

#2.모델
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

model = Sequential()
model.add(Dense(1000,input_dim=1))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1))

#3.컴파일
# from tensorflow.python.keras.optimizer_v1 import Adam, Adadelta, Adagrad, Adamax,RMSprop,SGD,Nadam
from tensorflow.python.keras.optimizer_v2 import adam,adamax,adadelta,adagrad,rmsprop,nadam
# from tensorflow.python.keras.optimizer_v1 import adam,adamax,Adam
# from keras import optimizers

learning_rate = 0.01
# optimizer = Adadelta(learning_rate=learning_rate)
# optimizer = Adagrad(learning_rate=learning_rate)
# optimizer = Adamax(learning_rate=learning_rate)
# optimizer = RMSprop(learning_rate=learning_rate)
# optimizer = SGD(learning_rate=learning_rate)
# optimizer = Nadam(learning_rate=learning_rate)
# optimizer = adam.Adam(lr=learning_rate)
optimizer = adadelta.Adadelta(lr=learning_rate)
# optimizer = rmsprop.RMSProp(lr=learning_rate)
# optimizer = adagrad.Adagrad(lr=learning_rate)
# optimizer = nadam.Nadam(lr=learning_rate)
# optimizer = adamax.Adamax(lr=learning_rate)

# learning_rate = 0.00001
# adamax loss 2.6148 lr 1e-05 result [[11.22882]]
# nadam loss 2.5864 lr 1e-05 result [[11.213009]]
# adagrad loss 5.3788 lr 1e-05 result [[8.386871]]
# rmsprop loss 2.5829 lr 1e-05 result [[11.415893]]
# adadelta nloss 42.0205 lr 1e-05 result [[0.2204634]]
# adam loss 2.594 lr 1e-05 result [[11.0852]]

# loss 2.5895 lr 0.0001 result [[11.18942]]
# loss 2.5748 lr 0.01 result [[11.154524]]

# opt = [adam,adamax,adadelta,adagrad,rmsprop,nadam]
# opti = lambda x: opt.


from tensorflow.python.keras.optimizer_v2 import adam,adadelta,adagrad,adamax,rmsprop,nadam

learning_rate = [0.1,0.01,0.001,0.0001,0.00001,0.5,0.05,0.005,0.0005]
for lr in learning_rate:
	op1 = adam.Adam(lr=lr)
	op2 = adadelta.Adadelta(lr=lr)
	op3 = adagrad.Adagrad(lr=lr)
	op4 = adamax.Adamax(lr=lr)
	op5 = rmsprop.RMSProp(lr=lr)
	op6 = nadam.Nadam(lr=lr)

	ops = [op1,op2,op3,op4,op5,op6]

	for opt in ops:
		model.compile(loss='mse',optimizer=opt)
		model.fit(x,y,epochs=50)
		loss = model.evaluate(x,y)
		pred = model.predict([11])
		opt_name = optimizer.__class__.__name__
		print(round(loss,4),lr,opt_name,pred,sep=',')
 

