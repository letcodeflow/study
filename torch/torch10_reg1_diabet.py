from pickletools import optimize
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test =  train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)
print(x_train.shape, y_train.shape)
x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test).unsqueeze(-1)
# Scaler

# x_train = x_train.reshape(16512, 4, 2) # 데이터의 갯수자체는 성능과 큰 상관이 없을 수 있다
# x_test = x_test.reshape(4128, 4, 2)
model = nn.Sequential(
    nn.Linear(10,100),
    nn.ReLU(),
    nn.Linear(100,150),
    nn.ReLU(),
    nn.Linear(150,50),
    nn.ReLU(),
    nn.Linear(50,10),
    nn.ReLU(),
    nn.Linear(10,1),
    nn.ReLU(),
)
# 2. 모델구성
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=23)
def train(model,criterion,optimizer,x,y):
    optimizer.zero_grad()
    hypothesis = model(x)
    loss = criterion(hypothesis,y)
    loss.backward()
    optimizer.step()

def evaluate(model,criterion,x,y):
    model.eval()
    with torch.no_grad():
        pred = model(x)
        loss = criterion(pred,y)

epochs = 100
for epoch in range(epochs):
    _ = train(model,criterion,optimizer,x_train,y_train)
pred = model(x_test)
print(pred)
print(y_test)
print(pred.shape,y_test.size())
print(r2_score(pred.detach(),y_test))

# 3. 컴파일, 훈련
# model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])
# earlyStopping = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1, restore_best_weights=True)
# hist = model.fit(x_train, y_train, epochs=200, batch_size=50,
#                 callbacks=[earlyStopping],
#                 validation_split=0.25)

# # 4. 평가, 예측
# loss = model.evaluate(x_test, y_test)
# print("loss : ", loss)

# y_predict = model.predict(x_test)
# r2 = r2_score(y_test, y_predict)
# print('r2스코어 : ', r2)

# loss :  [0.5216111540794373, 0.546445369720459]
# r2스코어 :  0.625834467060862

# lstm
# loss :  [0.3997993469238281, 0.44844144582748413]
# r2스코어 :  0.7132133703245158