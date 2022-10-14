#dacon 데이터 불러와서 답안지 작성

#1.데이터
#트레인, 테스트셋 불러오기

from sklearn.model_selection import train_test_split

#x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle = False)
#트레인 사이즈 및 변수표 정의, xy 불러와야함

#x = pd.read_csv(path + 'train.csv', index_col=0)
#판다스 read csv 함수 
import pandas as pd
# x = pd.read_csv(path + 'train.csv', index_col=0)
#트레인 셋을 먼저 불러온 후 해당 트레인 셋을 x .y로 나눠야함
#path 설정해야함
path = './_data/ddareunge/'
#x = pd.read_csv(path + 'train.csv', index_col=0)
#x로 나눌 트레인셋부터 설정

train_set = pd.read_csv(path + 'train.csv', index_col=0)
#트레인셋불러오고 인덱스는 삭제, 해당부분에서 count 제외하면 x, count는 y train

#count만 제거하는 과정이 필요함

print(train_set.shape)
#(1459, 10)
#ediccsv로 colum 확인
#info나 describe 사용
print(train_set.info())
#non-null값
print(train_set.describe())
#test셋 설정한 후 트레인셋과 육안비교
test_set = pd.read_csv(path + 'test.csv', index_col=0)
print(test_set.describe())
#count값이 y임을 확인

#그래서
x = train_set.drop(['count'], axis=1)
#트레인셋에서 count 제거, 해당 열을 아예 제거해야하므로 제거할 열 갯수 명시 
y = train_set['count']
#는 트레인셋에서 y만

#dims 확인
print(x.shape)

#null값
train_set.isnull().sum()
# 더해서 확인
print(train_set.isnull().sum())

#null값 제거
train_set = train_set.dropna()

#재확인
print(train_set.isnull().sum())

#제거하고나서 정의
x = train_set.drop(['count'], axis=1)
y = train_set['count']

#x,y 정의됐으므로 xy를 이용한 트레인셋과 테스트셋 설정
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle = False)
#사이킷런 모델셀렉션

#2.모델구성
model = Sequential()
#시퀀셜, 덴스 불러와야함

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(100, input_dims=9))
model.add(Dense(200))
model.add(Dense(200))
model.add(Dense(200))
model.add(Dense(200))
model.add(Dense(1))
#모델구성 y값은 1열이므로 
#dims 확인해야함

#3. 컴파일 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=4500, batch_size=200, verbose=1)
#오차식 mse, 최적화툴 아담, x트레인과 y트레인이용 4500회, 노드에 한번에 들어가능 스칼라수 200,출력형식 verbose1

#4.평가 예측
loss = model.evaluate(x_test, y_test)
#트레인셋에서 떨어져나온 count값이 있는 x,y를 넣어서 실제로 얼마나 일치하는지 로스율을 따진다
print('loss', loss)
#예측
y_predict = model.predict(x_test)
# 해당 가중치에 트레인셋에서 나온 x테스트값을 넣어서 (답안이 이미 있지만) y값을 만들어본다
#출력할 필요는 없고 r2스코어로 비교해본다
from sklearn.metrics import r2_score
r2=r2_score(y_test, y_predict)
#count값이 존재하는 y_test와 예측치로 나온 y_predict값을 비교해 스코어를 출력한다
print('r2', r2)

#5. csv불러와서 넣기

submission = pd.read_csv(path + 'submission.csv')
#함수 하나를 정의하고 해당파일에 답안지파일을 불러온다
#답안지는 id 인덱스와 count열로 이루어져있다.
#그러므로 정의된 함수에 y 프레딕트값을 채워준다
submission.

#6. rmse값 만들기





