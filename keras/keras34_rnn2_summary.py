
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,SimpleRNN, Dropout

#1.데이터구성
data = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6,],[5,6,7], [6,7,8],[7,8,9]])
y = np.array([4,5,6,7,8,9,10])

# x_shape(행, 열, 몇개씩 자르는지)
print(x.shape, y.shape)
x = x.reshape(7,3,1)
print(x.shape, y.shape)

model = Sequential()
model.add(SimpleRNN(100, input_shape=(5,1))) #공식문서 batch, timesteps, features
# biases = num_units*1
# num_units* num_units + num_features* num_units + biases
# unit*(feature+bias+unit)
#dnn unit*(feature+bias)
model.add(Dense(5, activation='relu'))
model.add(Dense(1))
model.summary()

""" 
model.compile(loss = 'mse', optimizer= 'adam')
model.fit(x,y, epochs=100
          )

# 평가예측
print(model.evaluate(x,y))
y_pred = np.array([8,9,10]).reshape(1,3,1)
print(y_pred)
result = model.predict(y_pred)
print(result) # [[[8]],[9],[10]]]
# np array타입인 8910을 1,3,1로 리쉐잎하겠다
 """