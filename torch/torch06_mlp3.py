import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
           [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
           [9,8,7,6,5,4,3,2,1,0]]
           )
y = np.array([11,12,13,14,15,16,17,18,19,20])

print(x.shape, y.shape)
x = np.transpose(x)
print(x.shape)
z = ([10,1.4,0])

x = torch.FloatTensor(x)
y = torch.FloatTensor(y).unsqueeze(-1)
z = torch.FloatTensor(z)

z = (z-torch.mean(x))/torch.std(x)
x = (x-torch.mean(x))/torch.std(x)

print(y.shape,z.shape)

model = nn.Sequential(
    nn.Linear(3,10),
    nn.Linear(10,100),
    nn.Dropout(0.3),
    nn.ReLU(),
    nn.Linear(100,1)
)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr = 0.0001)

def train(model, criterion, optimizer, x,y):
    optimizer.zero_grad()
    hypothesis = model(x)
    loss = criterion(hypothesis,y)
    
    loss.backward()
    optimizer.step()

def evaluate(model, criterion,x,y):
    model.eval()
    with torch.no_grad():
        pred = model(x)
        result = criterion(pred,y)
    return result

epochs = 20
for epoch in range(1,epochs+1):
    loss = train(model,criterion,optimizer,x,y)
    print(epoch, loss)

loss2 = evaluate(model,criterion,x,y)
print(loss2)
pred = model(z)
print(pred)