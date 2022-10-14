import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model  import Perceptron 
x = [[0,0],[0,1],[1,0],[1,1]]
y = [0,1,1,0]

model = SVC()

model.fit(x,y)

y_pred = model.predict(x)
print(x, y)
print(model.score(x,y))
print(accuracy_score(y,y_pred))

