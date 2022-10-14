import torch
import numpy as np
print(torch.__version__)
import torch.nn as nn #뉴럴 네트웍
import torch.optim as optim
import torch.nn.functional as F 
print(torch.cuda.is_available())

#1. data
x = np.array([1,2,3])
y = np.array([1,2,3])

#tortch tensor형태
#데이터가 (3, )형태일때 오류나는 경우가 있으므로 (3,1) 로 바꿔준다
x = torch.FloatTensor(x).unsqueeze(1) #1번째 자리에 스칼라 추가 
y = torch.FloatTensor(y).unsqueeze(-1) #같다
print(x,y)
print(x.shape, y.shape)

#2.model
model = nn.Sequential(
    nn.Linear(1,1), #input, output
    # nn.Conv1d(1,1,1),
    nn.Linear(1,1),
    nn.Linear(1,1),

)
#compile
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr=0.01) #모델에 옵티마이저를 써달라
optim.Adam(model.parameters(),lr=0.01) 
#model.compile(loss = 'mse',optimizer = 'SGD')


def train(model, criterion, optimizer, x,y):
    # model.train() 생략
    optimizer.zero_grad() #손실함수 기울기 초기화
    hypothesis = model(x)
    loss = criterion(hypothesis,y)
    loss.backward() #역전파하겠다
    optimizer.step() #역전파한 가중치를 갱신하겠다 = 1epoch

    return loss.item() #torch형태에서 알아볼수 있게

epochs = 1000
for epoch in range(1, epochs+1):
    loss = train(model,criterion,optimizer,x,y)
    print('epoch:{}, loss: {}'.format(epoch,loss))


#4 평가 예측
#loss = model.predict(x,y)

def evaluate(model, criterion, x,y): #가중치갱신이 필요없기때문에 옵티마이저없다 x_test,y_test
    model.eval() #평가모드
    with torch.no_grad(): #아래 모두 적용
        pred = model(x)    
        result = criterion(pred,y)
    return result.item()

loss2 = evaluate(model, criterion, x, y)
print('last loss', loss2)

#예측
# predict = model.predict([4])
result = model(torch.Tensor([[4]]))
print('4 예측값', result.item())