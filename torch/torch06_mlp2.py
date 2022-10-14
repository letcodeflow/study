import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#1. 데이터
x = np.array([range(10), range(21, 31), range(201, 211)])
#print(x)
# [[  0   1   2   3   4   5   6   7   8   9]
#  [ 21  22  23  24  25  26  27  28  29  30]
#  [201 202 203 204 205 206 207 208 209 210]]

print(x.shape) #(3, 10)
x= np.transpose(x)
print(x.shape) #(10, 3)
x = torch.FloatTensor(x)
y = np.array([[1,2,3,4,5,6,7,8,9,10],
           [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]]
           )
y = np.transpose(y)
y = torch.FloatTensor(y)
print(y.shape)

z = ([9, 30, 201])
z = torch.FloatTensor(z)

model=nn.Sequential(
    nn.Linear(3,40),
    nn.Linear(40,100),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(100,2)
)


optimizer = optim.Adam(model.parameters(),lr = 0.001 )
criterion = nn.MSELoss()

def train(model, criterion, optimizer, x,y):
    optimizer.zero_grad()
    hypothesis = model(x)
    loss = criterion(hypothesis,y)
    loss.backward()
    optimizer.step()
    return loss

epochs =100
for epoch in range(1,epochs+1):
    loss = train(model, criterion,optimizer,x,y)
    if epoch % 10 ==0:
        print(epoch,'\n',loss)  

def evaluate(model,criterion,x,y):
    model.eval()
    
    with torch.no_grad():
        pred = model(x)
        result = criterion(pred,y)
    return result

lossa = evaluate(model,criterion,x,y)
print('lasst loss',loss)
result = model(z)
print(result)