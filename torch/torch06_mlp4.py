import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

#1. 데이터
x = np.array([range(10)])
x= np.transpose(x)
print(x.shape) #(10, 1)
x = torch.FloatTensor(x)



y = np.array([[1,2,3,4,5,6,7,8,9,10],
           [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
           [9,8,7,6,5,4,3,2,1,0]]
           )
y = np.transpose(y)
print(y.shape)
y = torch.FloatTensor(y)
print(y.shape)
z = np.array([[9]])
z = torch.FloatTensor(z)

z = (z-torch.mean(x))/torch.std(x)
x = (x-torch.mean(x))/torch.std(x)
#2.모델
model = nn.Sequential(
    nn.Linear(1,10),
    nn.Linear(10,4),
    nn.ReLU(),
    nn.Linear(4,3)
)
# optimizer = 
optimizer = optim.Adam(model.parameters(),lr=0.01)
criterion = nn.MSELoss()
def train(model, criterion,optimizer, x,y):
    optimizer.zero_grad()
    hypothesis = model(x)
    loss = criterion(hypothesis,y)
    loss.backward()
    optimizer.step()
    return loss.item()


epochs = 200
for epoch in range(1, epochs+1):
    loss = train(model,criterion,optimizer,x,y)
    print(epoch, loss, sep='\n')

def evaluate(model, criterion, x,y):
    model.eval()
    with torch.no_grad():
        pred = model(x)
        result = criterion(pred,y)
    return result.item()

loss2 = evaluate(model, criterion, x,y)
print(loss2)

result = model(z)
print(result)