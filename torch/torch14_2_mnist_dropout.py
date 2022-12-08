from turtle import forward
from unittest import TestLoader
from torchvision.datasets import MNIST
from torch.utils.data import TensorDataset,DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.transforms as tr
transf = tr.Compose([tr.Resize(15), tr.ToTensor()]) #구성, 
path = './_data/torch_data/MNIST/'

train = MNIST(path, train=True, download=True, transform=transf)
test = MNIST(path, train=False, download=True, transform=transf)
# train = MNIST(path, train=True, download=False, )
# test = MNIST(path, train=False, download=False, )
# print(train[0][0].shape)

x_train, y_train = train.data/255., train.targets
x_test, y_test = test.data/255., test.targets


print(y_train.shape, x_test.shape)

print(np.min(x_train.numpy()), np.max(x_train.numpy()))

x_train, x_test = x_train.view(-1,x_train.size()[1]*x_train.size()[2]),x_test.view(-1,x_train.size()[1]*x_train.size()[2])
print(x_train.size())

# train_set = TensorDataset(x_train,y_train)
# test_set = TensorDataset(x_test,y_test).

train_loader = DataLoader(train, batch_size=60, shuffle=True)
teste_loader = DataLoader(test, batch_size=600, shuffle=False)

class DNN(nn.Module):
    def __init__(self, num_features):
        super(DNN, self).__init__()
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(num_features, 100),
            nn.Dropout(0.2),
            nn.ReLU(),
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
        )
        self.output_layer3 = nn.Sequential(
            nn.Linear(50, 10),
            # nn.Softmax(),
        )

    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.output_layer3(x)

        return x

model = DNN(28*28)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=1e-4)

def train(model, criterion, optimizer, loader):
    epoch_loss = 0
    epoch_acc = 0
    for batch_x, batch_y in loader:
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
            pred = model(x_test_batch)
            loss = criterion(pred, y_test_batch)
        
        
            epoch_loss += loss.item()
            y_pred = torch.argmax(pred,1)
            acc= (y_pred == y_test_batch).float().mean()
            acc += acc.item()
        return epoch_loss/len(loader), acc/len(loader)
epochs = 20
for epoch in range(epochs):
    loss, acc = train(model,criterion,optimizer,train_loader)
    val_loss, val_acc = evaluate(model, criterion, teste_loader)
    if epoch % 2 ==0:
        
        print(epoch,'\n', loss,'\n',acc, '\n', val_loss,'\n',val_acc)





