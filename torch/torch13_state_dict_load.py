from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
torch.manual_seed(23)


USE_CUDA = torch.cuda.is_available
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu') # ['cuda:0', 'cuda:1'] 2개 이상일때는 list

data = load_breast_cancer()
x = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.78, stratify=y, random_state=342)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(-1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(-1).to(DEVICE)

from torch.utils.data import TensorDataset,DataLoader

train = TensorDataset(x_train,y_train)
test = TensorDataset(x_test,y_test)

# print(train[0][0])

train_loader = DataLoader(train,batch_size=40,shuffle=True)
test_loader = DataLoader(test,batch_size=40,shuffle=True)

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
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
path = './save'
print(model)
loaded_model = Model(30,1).to(DEVICE)
loaded_model.load_state_dict(torch.load(path +'torch13_state_dict.pt'))

y_predict = (loaded_model(x_test) >= 0.5).float() # (model(x_test) 이거 다시 질문 해야한다 !!!!!!!!!!!!!!!!!!!!!
print(y_predict[:10])

score = (y_predict == y_test).float().mean() # 0, 1 개수 가지고 평균을 낸것이 accuracy
print('accuracy : {:.4f}'.format(score))

from sklearn.metrics import accuracy_score
# score = accuracy_score(y_test, y_predict) # gpu상태니까 cpu로 바꿔야 한다
score = accuracy_score(y_test.cpu(), y_predict.cpu())
# score = accuracy_score(y_test.cpu().numpy(), y_predict.cpu().numpy())
print('accuracy :', score) 
""" 
criterion = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, criterion, optimizer, loader):
    total_loss = 0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        loss = criterion(hypothesis,y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, criterion, loader):
    model.eval()
    total_loss = 0
    for x_batch, y_batch in loader:
        with torch.no_grad():
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            total_loss += loss.item()
    return total_loss/len(loader)

for epoch in range(100):
    loss = train(model, criterion, optimizer, train_loader)
    print('epoch:{},loss:{}'.format(epoch, loss))

loss = evaluate(model, criterion, test_loader)

y_predict = (model(x_test) >= 0.5).float() # (model(x_test) 이거 다시 질문 해야한다 !!!!!!!!!!!!!!!!!!!!!
print(y_predict[:10])

score = (y_predict == y_test).float().mean() # 0, 1 개수 가지고 평균을 낸것이 accuracy
print('accuracy : {:.4f}'.format(score))

from sklearn.metrics import accuracy_score
# score = accuracy_score(y_test, y_predict) # gpu상태니까 cpu로 바꿔야 한다
score = accuracy_score(y_test.cpu(), y_predict.cpu())
# score = accuracy_score(y_test.cpu().numpy(), y_predict.cpu().numpy())
print('accuracy :', score) 



path = './save'
torch.save(model.state_dict(), path+'torch13_state_dict.pt')
 """