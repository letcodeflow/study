from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn
import torch.nn as nn
import torch.optim as optim


USE_CUDA = torch.cuda.is_available
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

datasets = load_breast_cancer()
x = datasets.data
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7, random_state=24, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)

y_train = torch.FloatTensor(y_train).unsqueeze(-1).to(DEVICE) #iris와 다르게 왜 언스퀴즈를 쓰고 LongTensor가 아닌 FloatTnesorf를 썼는지
y_test = torch.FloatTensor(y_test).unsqueeze(-1).to(DEVICE)

#2.model
class Model(nn.Module):
    def __init__(self,input_dim, output_dim):
        # super().__init__()
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dim,64)
        self.linear2 = nn.Linear(64,32)
        self.linear3 = nn.Linear(32,16)
        self.linear4 = nn.Linear(16,output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_size):
        x = self.linear1(input_size)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.sigmoid(x)
        return x

model = Model(30,1).to(DEVICE)

criterion = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr = 0.01)

def train(model, criterion, optimizer, x_train, y_train):
    optimizer.zero_grad()
    hypothesis = model(x_train)
    loss = criterion(hypothesis, y_train)

    loss.backward()
    optimizer.step()

    return loss.item()

Epochs = 100

for epoch in range(Epochs):
    loss = train(model,criterion,optimizer,x_train,y_train)
    print('epoch:{},loss:{}'.format(epoch,loss))

def evaluate(model, criterion, x,y):
    model.eval()
    with torch.no_grad():
        pred = model(x)
        loss = criterion(pred,y)
        return loss.item()

loss = evaluate(model,criterion,x_test,y_test)

y_pred = (model(x_test) >=0.5).float()
print(y_pred[:10])

score = (y_pred==y_test).float().mean()
print(score.item())