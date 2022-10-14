from turtle import forward
from unittest import TestLoader
from torchvision.datasets import CIFAR100
from torch.utils.data import TensorDataset,DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.transforms as tr
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
transf = tr.Compose([tr.Resize(15), tr.ToTensor()]) #구성, 
path = './_data/torch_data/CIFAR100/'

train = CIFAR100(path, train=True, download=True, transform=transf)
test = CIFAR100(path, train=False, download=True, transform=transf)
# train = MNIST(path, train=True, download=False, )
# test = MNIST(path, train=False, download=False, )
# print(train[0][0].shape)

x_train, y_train = train.data/255., train.targets
x_test, y_test = test.data/255., test.targets
x_train, x_test = torch.FloatTensor(x_train), torch.FloatTensor(x_test)
y_train, y_test = torch.LongTensor(y_train), torch.LongTensor(y_test)

x_train, x_test = x_train.reshape(-1,3,32,32), x_test.reshape(-1,3,32,32)
print(y_train.shape, x_test.shape)

# print(np.min(x_train.numpy()), np.max(x_train.numpy()))

# 좌표값 tensorflow 60000 28 28 1 torch 60000 1 28 28
# x_train, x_test = x_train.view(-1,x_train.size()[1]*x_train.size()[2]),x_test.view(-1,x_train.size()[1]*x_train.size()[2])
# x_train, x_test = x_train.unsqueeze(1),x_test.unsqueeze(1)
print(x_train.size())

train_set = TensorDataset(x_train,y_train)
test_set = TensorDataset(x_test,y_test)

train_loader = DataLoader(train_set, batch_size=60, shuffle=True)
teste_loader = DataLoader(test_set, batch_size=600, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.hidden_layer1 = nn.Sequential(
            nn.Conv2d(3,64, kernel_size=(3,3), stride=1),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
        )
        self.hidden_layer1_1 = nn.Sequential(
            nn.Conv2d(64,32, kernel_size=(3,3),),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
        )
        self.flatten_1 = nn.Linear(32*6*6, 300)

        self.hidden_layer2 = nn.Sequential(
            nn.Linear(300, 150),
            nn.ReLU(),
        )
        self.output_layer3 = nn.Sequential(
            nn.Linear(150, 100),
            # nn.Softmax(),
        )

        # self.output_layer4 = nn.Linear(in_features=50, out_features=10)

    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer1_1(x)
        x = x.view(x.size(0), -1) #=flatten
        x = self.flatten_1(x)
        x = self.hidden_layer2(x)
        x = self.output_layer3(x)

        return x

model = CNN().to(DEVICE)


from torchsummary import summary
summary(model, (3,32,32))
# print(model) #model summary
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=1e-4)

def train(model, criterion, optimizer, loader):
    epoch_loss = 0
    epoch_acc = 0
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
        optimizer.zero_grad()
        hypothesis = model(batch_x)
        loss = criterion(hypothesis, batch_y)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        y_pred = torch.argmax(hypothesis,1)
        acc= (y_pred == batch_y).float().mean()
        acc += acc.item()
    return epoch_loss/len(loader), acc/len(loader)
    #hist = model.fit(x_train,y_train)

def evaluate(model, criterion, loader):
    model.eval() #레이어 단계 배치놈 드롭아웃 등 미적용
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for x_test_batch, y_test_batch in loader:
            x_test_batch, y_test_batch = x_test_batch.to(DEVICE), y_test_batch.to(DEVICE)
            pred = model(x_test_batch)
            loss = criterion(pred, y_test_batch)
        
        
            epoch_loss += loss.item()
            y_pred = torch.argmax(pred,1)
            acc= (y_pred == y_test_batch).float().mean()
            acc += acc.item()
        return epoch_loss/len(loader), acc/len(loader)
epochs = 1
for epoch in range(epochs):
    loss, acc = train(model,criterion,optimizer,train_loader)
    val_loss, val_acc = evaluate(model, criterion, teste_loader)
    if epoch % 2 ==0:
        
        print(epoch,'\n', loss,'\n',acc, '\n', val_loss,'\n',val_acc)




