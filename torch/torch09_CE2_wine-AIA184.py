import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.datasets  import load_wine

#1.데이터
datasets = load_wine()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)
print(np.unique(y))
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, shuffle=True, random_state=137)
x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)
model = nn.Sequential(
    nn.Linear(13,26),
    nn.ReLU(),
    nn.Linear(26,38),
    nn.ReLU(),
    nn.Linear(38,13),
    nn.Linear(13,3),
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

from sklearn.metrics import accuracy_score
def train(model,criterion,optimizer,x,y):
    optimizer.zero_grad()
    hypothesis = model(x)
    loss = criterion(hypothesis,y)
    loss.backward()
    optimizer.step()
    
def evaluate(model,criterion,x,y):
    model.eval()
    with torch.no_grad():
        pred = model(x)
        loss = criterion(pred,y)

epochs = 10
for epoch in range(epochs):
    loss = train(model,criterion,optimizer,x_train,y_train)
    
loss2 = evaluate(model,criterion,x_test,y_test)
pred = model(x_test)
print(accuracy_score(torch.argmax(pred,1),y_test))
    
    
    
#3.컴파일 훈련
# model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# hist = model.fit(x_train, y_train, epochs=1, verbose=1, 
#           validation_split=0.2,
#           callbacks=ES)

# #4.평가 예측
# loss = model.evaluate(x_test, y_test)

# y_predict = model.predict(x_test)
# # y_predict = model.predict(x_test[:5])
# print(y_test)
# print(y_predict)

# y_predict = np.argmax(y_predict, axis= 1)
# y_test = np.argmax(y_test, axis= 1)
# print(y_test)
# print(y_predict)

# from sklearn.metrics import accuracy_score


# # y_predict = y_predict.round(0)
# # # pre2 = y_predict.flatten() # 차원 펴주기
# # # pre3 = np.where(pre2 > 1, 2 , 0) #1보다크면 2, 작으면 0
# acc = accuracy_score(y_test, y_predict)

# # print(y_predict)
# print('loss : ', loss[0])
# #loss식의 첫번째
# print('acc :',  loss[1])
# #loss식의 두번째
# print('acc', acc)

#민맥스
# loss :  0.24404345452785492
# acc : 0.9629629850387573
# acc 0.9629629629629629

#스탠
# loss :  0.12898489832878113
# acc : 0.9629629850387573
# acc 0.9629629629629629

#스케일미적용
# loss :  0.7572556138038635
# acc : 0.6481481194496155
# acc 0.6481481481481481

# dropout
# loss :  1.0873541831970215
# acc : 0.29629629850387573
# acc 0.2962962962962963

# lstm
# loss :  1.091688632965088
# acc : 0.40740740299224854
# acc 0.4074074074074074