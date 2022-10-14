from pickletools import optimize
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#1. 데이터
datasets = load_digits()
x_data = datasets.data
y_data = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,train_size=0.8,random_state=234)
print(x_data.shape, np.unique(y_data.shape))
x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

model = nn.Sequential(
    nn.Linear(64,128),
    nn.ReLU(),
    nn.Linear(128,48),
    nn.ReLU(),
    nn.Linear(48,32),
    nn.ReLU(),
    nn.Linear(32,10),
)

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
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)
epochs = 21
for epoch in range(epochs):
    _ = train(model, criterion,optimizer,x_train,y_train)

pred = model(x_test)
pred = torch.argmax(pred,1)
print(accuracy_score(pred,y_test))


# votind
# LogisticRegression 정확도: 0.9528
# KNeighborsClassifier 정확도: 0.9500
# XGBClassifier 정확도: 0.9611
# LGBMClassifier 정확도: 0.9639
# CatBoostClassifier 정확도: 0.9667


# x(1797, 64)
# y[0 1 2 3 4 5 6 7 8 9]

# lda 10
# 시간 1.3313465118408203
# 결과 0.9472222222222222

# pca 60
# 시간 1.965287446975708
# 결과 0.9444444444444444

# no pca lda
# 시간 1.5885100364685059
# 결과 0.9555555555555556

# xgb
# 시간 2.213334560394287
# 결과 0.9555555555555556

# xgb bagging100
# 시간 90.1886818408966
# 결과 0.9611111111111111