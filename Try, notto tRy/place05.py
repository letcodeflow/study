#데이터셋을 불러온다
from sklearn.datasets import fetch_california_housing
#불러왔으면 먼저 해당데이터셋에 이름을 부텽준다
datasets = fetch_california_housing()
#해당데이터에서 나뉘어 있는 주로 예측값/그외를 선별해준다

x = datasets.data
y = datasets.target #target 을 보통 예측하려고 한다

#해당하는 데이터의 와꾸를 조사한다
print(x.shape)
print(y.shape)

print(datasets.DESCR)
#특성의 정확한 설명과 몇개인지 확인한다

#평가할 데이터가 따로없으므로 학습용과 평가용 데이터로 나눈다
#데이터의 범위는 넓을수록 좋으므0로 적은 데이터셋을 랜덤하게 뽑아낸다
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)
#x와 y 데이터셋을 이용해 학습용 80% 평가용 20%로 구성한다. 실행때마다 달라진 난수표를 적용해
#결과값 비교가 어렵게 하는 것을 피하기 위해 난수표를 고정한다

#정제된 데이터이므로 모델구성
from tensorflow.keras.models import Sequential
#모델구성에는 모델에 사용되어질 구조와 레이어의 종류가 필요하다
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(5, input_dim=8)) #8개의 특성값을 통해 유추한다 모델에 Dense라는 층을 더해 노드는 5개=첫번째 아웃풋
model.add(Dense(5))#줄카피 컨트롤씨 줄삭제 쉬프트델 
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))
#마지막 레이어는 구하고 싶은 예측치의 종류 수에 따라 다르다 현재는 집값 하나

#3. 모델이 구성되었으므로 오차식와 최적화툴을 더하고 이를 훈련시킨다
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 100, batch_size = 100)
#x 학습용셋과 y 학습용데이터셋을 이용해 epoch 100회로 돌린다 한번에 들어가는 스칼라는 100이다

#4. 평가 예측
#훈련이 끝났으므로 오차율과 예측값을 구한다
loss = model.evaluate(x_test, y_test)
#앞선 모델구성에서 해당 모델에 w값을 저장하기로 하고 훈련이 되었으므로 현재 w값이 저장돼있다
#이를 evaluate 함수로 x, y 평가데이터셋을 넣어 오차율을 구한다
print(loss)
#예측하기
y_predict=model.predict(x_test)
#가중치 w에 x 평가셋을 넣어 y값이 어떻게 나오는지 예측한다

#결정계수
from sklearn.metrics import r2_score
#사이킷런 메트릭스에 있는 r2스코어를 불러와 예측된 y값과 평가용으로 남겨진 실제 y값을 비교하여
#오차율이 적용된 머신러닝의 식이 얼마나 부합하는지 숫자로 확인한다
r2= r2_score(y_test, y_predict)
print(r2)