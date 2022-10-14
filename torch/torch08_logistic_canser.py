#회귀모델에 sigmoid를 붙였다 
# from xml.etree.ElementTree import C14NWriterTarget
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings
warnings.filterwarnings(action='ignore')

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'epcu')

#1.data
datasets = load_breast_cancer()
x = datasets.data
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.87,random_state=33,stratify=y)
# x = torch.FloatTensor(x)
# y = torch.FloatTensor(y)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(-1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(-1).to(DEVICE)

model = nn.Sequential(
    nn.Linear(30,64),
    nn.ReLU(),
    nn.Linear(64,32),
    nn.ReLU(),
    nn.Linear(32,1),
    nn.Sigmoid(),
).to(DEVICE)


criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(),lr = 0.001)

def train(model,criterion,optimizer,x,y):
    optimizer.zero_grad()
    hypothesis = model(x)
    loss = criterion(hypothesis,y)
    loss.backward()
    optimizer.step()
    return loss

epochs = 201
for epoch in range(epochs):
    loss = train(model, criterion,optimizer,x_train,y_train)
    print(epoch,'\n',loss.item())

def evaluate(model, criterion ,x,y):
    model.eval()
    with torch.no_grad():
        pred  = model(x)
        loss = criterion(pred, y)
        
    return loss

loss = evaluate(model,criterion,x_test,y_test)

# print(pred)
    #  [9.9983e-01],
    #     [1.1834e-07],
    #     [9.8953e-01],
    #     [2.1824e-07],
    #     [1.0665e-04],
    #     [9.2698e-02],
    #     [7.9681e-04]], device='cuda:0', grad_fn=<SigmoidBackward0>)
#0부터 1사이, cuda 붙어있고 grad

print(loss.item())
pred = (model(x_test)>=0.5).float()
# score = (pred ==y_test).float().mean()
# print(score.item())

from sklearn.metrics import accuracy_score
score = accuracy_score(pred.cpu(), y_test.cpu())