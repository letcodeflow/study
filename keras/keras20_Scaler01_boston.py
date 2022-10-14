from sklearn import metrics
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler #대문자 클래스 약어도 ㄷ대문자로
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

data_sets = load_boston()
import numpy as np
x = data_sets.data
y = data_sets.target
# print(np.min(x))
# print(np.max(x))
# # 0.0
# # 711.0
# x = (x-np.min(x))/((np.max(x)-np.max(x)))
# print(x[:10])
# #0과 1사이로 수렴
# print(x.shape, y.shape)
#전체 컬럼에 수행한 스케일러이므로 주석처리한다




from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=False)
scaler = MinMaxScaler()
# scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
#스케일러정의
scaler.fit(x_train)
#핏한다
x_train = scaler.transform(x_train)
#컬럼별로 스케일 해야하는데 이것을 함수정의자들이 예상하지 않았을까?
#수치로 바꾼것을 본래 데이터셋으로 재정의한다
x_test = scaler.transform(x_test)
#x_test를 트레인에 맞춰서 변환한 값으로 변환한다 - 트레인으로 피트한 값을 스케일러가 가지고 있고 그것으 ㄹ기준으로 트랜스폼하기 때문
#이부분이 이해가 안됨
print(np.min(x_train))
print(np.max(x_train))
# 0.0
# 1.0
print(np.min(x_test))
print(np.max(x_test))
# -0.21613002146580804
# 1.0939457202505218
a = 0.1
b = 0.2
print(a +b)
# 0.30000000000000004

#스케일러 하기전
#민맥스
#스탠다드


print(x_train)
print(x_test)
print(y_train)
print(y_test)
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

print(x.shape)
model = Sequential()
model.add(Dense(5, input_dim=13))
model.add(Dense(5, activation='sigmoid'))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='sigmoid'))
model.add(Dense(5))
model.add(Dense(1))

#컴파일 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics=['accuracy', 'mae'])
#컴파일 전후 얼리스타핑 미니멈 혹은 맥시멈값을 patience 지켜보고 있다가 정지시키는 함수

from tensorflow.python.keras.callbacks import EarlyStopping
ES = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=2, restore_best_weights=True)

#훈련을 인스턴스 하나로 줄여주기
h = model.fit(x_train, y_train, epochs=100, batch_size=10,
          validation_split=0.1, 
          callbacks=[ES], 
          verbose=3)

#훈련돼ㅆㅇ니 평가 예측

loss = model.evaluate(x_test, y_test)
print('loss', loss)

print(h) #훈련 h가 어느 메모리에 저장돼있는가
print('=======================================')
print(h.history) #훈련 h의 loss율 그리고 추가돼있다면 validation loss 율\

y_predict = model.predict(x_test)

#y예상값이 나왔으므로 실제 y값과 비교해본다
from sklearn.metrics import r2_score
r2= r2_score(y_test, y_predict)
print('r2', r2)
print('loss', loss)

#그리기

import matplotlib.pyplot as plt#맷플롯라이브러리의 파이플랏
plt.figure(figsize=(4,5)) #표의 크기 마우스로 조절가능
plt.plot(h.history['loss'], marker = '.', c='red', label='loss')
#맷플롯립을 이용해 그리겠다. 훈련값h 딕셔너리에 있는 loss 키값을 불러와서 .을 찍고 컬러는 레드, 그래프이름은 loss
plt.plot(h.history['val_loss'], marker = '_', c='blue', label='val_loss')
#맷플랏라이브러리를 이용해 그리겠다. plot 함수를 이용하겠다
# 훈련 로스율 딕셔너리 h의 valoss 키를 이용하여 _로 점을 찍고 로스그래프와 구분하기 위해
#파란색, 그래프 이름은 발로스
plt.grid()#모눈종이
plt.title(1847)#그래프 이름 한글위해선 코드 따와야함, 인수 있어야함
#plt.xlabel('epochs')# x축을 epochs라 명하겠다
plt.ylabel('loss') #y축을 loss라 명하겠다
plt.legend(loc='upper left') #그래프이름 표기를 위쪽왼쪽에 위치시키겠다(인수는 생략가능, 띄어쓰기= 그래프 빈자리에 생성됨)
plt.show()#보여달라

#vali 미적용
# r2 -0.5474061321360475
# loss [41.60723114013672, 0.0, 5.335869312286377]

# #적용
# r2 0.05849269602291585
# loss [25.31559944152832, 0.0, 4.08815336227417]

# 사이즈가 작은데도 상당히 큰 효과가 있었다. 리니어모델이라서 그런가?


#minmax
# r2 -0.11763746977833955
# loss [30.051448822021484, 0.0, 4.538322925567627]

#standard
# r2 0.1466862160990129
# loss [22.944215774536133, 0.0, 3.8606574535369873]
