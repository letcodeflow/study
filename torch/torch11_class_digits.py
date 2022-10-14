from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available
DEVICE = ('cuda' if USE_CUDA else 'cpu')
#1.data
data = load_digits()
x = data.data
y = data['target']

x_train, x_test, y_train,y_test = train_test_split(x,y,random_state=23,train_size=0.5,stratify=y)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.LongTensor(y_train).to(DEVICE)
y_test = torch.LongTensor(y_test).to(DEVICE)
print(y_test.shape)
#2.model
class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dim,64)
        self.linear2 = nn.Linear(64,32)
        self.linear3 = nn.Linear(32,16)
        self.linear4 = nn.Linear(16,output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, input_size):
        x = self.linear1(input_size)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.softmax(x)
        return x

model = Model(64,10).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x,y):
    optimizer.zero_grad()
    hypothesis = model(x)
    loss =criterion(hypothesis,y)
    loss.backward()
    optimizer.step()
    return loss.item()

epochs = 100
for epoch in range(epochs):
    loss = train(model, criterion,optimizer,x_train,y_train)
    print('epoch:{},loss:{}'.format(epoch,loss))

def evaluate(model,criterion,x,y):
    model.eval()
    with torch.no_grad():
        pred = model(x)
        loss = criterion(pred,y)
        
        return loss.item()

loss = evaluate(model,criterion,x_test,y_test)

y_pred = model(x_test)
# print(y_pred)

score = (torch.argmax(y_pred,1) == y_test).float().mean()
print(score)

