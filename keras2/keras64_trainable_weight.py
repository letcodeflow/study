import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1.data
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

#2.model
model = Sequential()
model.add(Dense(3,input_dim=1))
model.add(Dense(2))
model.add(Dense(1))


print(len(model.weights))
print(len(model.trainable_weights))


model.trainable = False
print(len(model.weights))
print(len(model.trainable_weights))

model.summary()

model.compile(loss = 'mse',optimizer='adam')

model.fit(x,y,batch_size=1,epochs=100)

y_pred = model.predict(x)
print(y_pred[:3])