import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.linear_model  import Perceptron 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from torch import sigmoid

x = [[0,0],[0,1],[1,0],[1,1]]
y = [0,1,1,0]

model = Sequential()
model.add(Dense(1, input_dim=(2),activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam')
model.fit(x,y, epochs=100)

y_pred = model.predict(x)
print(x, y)
print(model.evaluate(x,y))
print(accuracy_score(y,y_pred))

