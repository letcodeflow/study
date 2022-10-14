from pickletools import optimize
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.datasets import fetch_covtype
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print(DEVICE)
print(torch.cuda.device_count())
#1.데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(y)
print(x.shape, y.shape)
print(np.unique(y, return_counts=True))
# (581012, 54) (581012,)
# (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],
#       dtype=int64))
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, shuffle=True, random_state=137)
x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.LongTensor(y_train).to(DEVICE)
y_test = torch.LongTensor(y_test).to(DEVICE)

model= nn.Sequential(
    nn.Linear(54,158),
    nn.ReLU(),
    nn.Linear(158,328),
    nn.Dropout(0.3),
    nn.Linear(328,77),
    nn.ReLU(),
    nn.Linear(77,7),
).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
def train(model,criterion, optimizer, x,y):
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

epochs = 19
for epoch in range(epochs):
    _ = train(model, criterion,optimizer,x_train,y_train)

pred = model(x_test)
pred = torch.argmax(pred,1)
from sklearn.metrics import accuracy_score
print(accuracy_score(pred,y_test))
#3.컴파일 훈련