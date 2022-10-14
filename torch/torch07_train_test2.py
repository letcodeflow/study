import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split




x = np.array([1,2,3,4,5,6,7,8,9,10]).T
y = np.array([1,2,3,4,5,6,7,8,9,10]).T
x_predict = np.array([11,12,13]).T


x = torch.FloatTensor(x).unsqueeze(-1)
y = torch.FloatTensor(y).unsqueeze(-1)
x_predict = torch.FloatTensor(x_predict).unsqueeze(-1)

print(x.shape,y.shape)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, #test_size=0.3, 
    train_size=0.7,
    #shuffle=False,
    random_state=66
)
model = nn.Sequential(
    nn.Linear(1,100),
    nn.Linear(100,10),
    nn.Linear(10,100),
    nn.Linear(100,10),
    nn.Linear(10,100),
    nn.Linear(100,10),
    nn.Linear(10,100),
    nn.Linear(100,1),
)
optimizer = optim.Adam(model.parameters(),lr=0.000000001)
criterion = nn.MSELoss()
def train(model, criterion, optimizer, x,y):
    # model.train()
    optimizer.zero_grad()
    hypothesis = model(x_train)
    loss = criterion(hypothesis, y)
    
    loss.backward()
    optimizer.step()

def evaluate(model, criterion, x,y):
    model.eval()
    with torch.no_grad():
        y_predict = model(x)
        result = criterion(y_predict,y)
    return result

epochs=21
for epoch in range(1,epochs):
    loss = train(model, criterion,optimizer,x_train,y_train)
    print(epoch,'\n',loss)

loss2 = evaluate(model,criterion,x_test,y_test)
print('최종로스',loss2.item())

prediction = model(x_predict)

print(prediction.cpu().detach().numpy())

#프레딕션을 item()으로 뽑을 수 없을대  detach를 붙여서 출력중 grad 부분을 떼어내고 .cpu 를 붙여서 gpu메모리에 있는 tensor를 cpu로 넘겨준다