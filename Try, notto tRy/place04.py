#1.데이터
#데이터타입 구현
import numpy as np

#데이터정의
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

#2. 모델구성
#딥러닝 수직구조이다
#input_dim값은 특성 컬럼 피쳐와 같고 마지막 레이어의 노드수는 추론하고자 하는 예측값의 수와 같다=y
from tensonrflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#모델에 쓸 레이어의 종류

#모델은 시퀀셜을 쓴다
model=Sequential()
#맨처음 특성을 포함해 인풋 아웃풋=머신러닝 구성+이후 수직구조의 딥러닝
model.add(Dense(5, input_dim=5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일 훈련
#구성된 모델에 들어갈 트레이닝 데이터셋과 로스식, 최적화툴, 횟수, 노드당 들어가는 배치사이즈를 구해준다

#2번에서 정의된  model을 어떻게 할 것이냐-> compile 적용
model.compile(loss = 'mse', optimizer='adam')
#컴파일하면서 오차율식과 최적화툴을 적용한다
#훈련
model.fit(x, y, epochs=100, batch_size=1)

#훈련이 끝났으므로 평가및 예측해본다

#4. 평가 예측
#평가는 오차율이 얼마나 나오는지
loss = model.evaluate(x, y)
#x,y 데이터를 넣어서 평가함수
print(loss)

예측
result = model.predict(x)
#x혹은 그 이외의 자료를 넣어 해당하는 y 예측값을 도출한다
print(result)