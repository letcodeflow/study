#목표 라벨인코더 원핫인코더 이해
#데이터전처리 적용

import tensorflow as tf
tf.random.set_seed(31)
path = './_data/kaggle_titanic/'
import imp
import pandas as pd
train_set = pd.read_csv(path + 'train.csv')
test_set = pd.read_csv(path + 'test.csv')
gender_submission = pd.read_csv(path + 'gender_submission.csv')
#불러오기 완료 
#라벨인코더 적용기준 문자열고유값으로 분류된 경우 해당 문자를 숫자에 대입

print(train_set['Pclass'].unique()) #판다스 유니크는 특정 열에 사용되는듯 
print(train_set['Embarked'].unique())
print(train_set.value_counts()) #데이터셋 전체를 넣으니 데이터전부가 일치하는 행이 있는지 표시하는 것 같다
#필요한것? object 타입 드롭중 Sex만 바꿔서 넣고 embarked 바꿔넣기를 시도해본다
#불러오기
print(train_set.shape)
print(test_set.shape)

tatu1 = pd.get_dummies(train_set['Sex'])
tatu2 = pd.get_dummies(test_set['Sex'])
#프리픽스를 붙여야 나눠지나? 아니다
train_set = pd.concat([train_set,tatu1], axis=1)
test_set = pd.concat([test_set,tatu2], axis=1)
#겟더미는 붙이는 과정이 필요하다 난값처리는 두가지 방법이 있는듯 하다 
#원핫인코딩은 argmax 겟더미는 컨캣이 필요한듯하다



print(train_set['Sex'])
print(train_set)


print(train_set.columns)
print(test_set.columns)
# Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
#        'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
#       dtype='object') 트레인
# Index(['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',
#        'Ticket', 'Fare', 'Cabin', 'Embarked'],
#       dtype='object') 테스트
# 테스트에 서바이브드 없는거 확인
print(train_set.shape)
print(test_set.shape)
# (891, 12)
# (418, 11)
#x,y 나누기위해서는 떨굴 열을 알아야함
print(train_set.describe()) # 평균치는 지금 피료없음
print(train_set.info())
#  0   PassengerId  891 non-null    int64
#  1   Survived     891 non-null    int64
#  2   Pclass       891 non-null    int64
#  3   Name         891 non-null    object
#  4   Sex          891 non-null    object
#  5   Age          714 non-null    float64
#  6   SibSp        891 non-null    int64
#  7   Parch        891 non-null    int64
#  8   Ticket       891 non-null    object
#  9   Fare         891 non-null    float64
#  10  Cabin        204 non-null    object
#  11  Embarked     889 non-null    object
# 패신저아이디는 인덱스컬럼이므로, 네임, 성별, 티켓, 캐빈, 엠바크드 삭제



#결측치 확인
print(train_set.isnull().sum())
print(test_set.isnull().sum())
#캐빈값을 결측치가 너무 많으므로 이미 삭제 age fare 결측치에서 난값발생? 일단 dropna로 드랍
train_set.dropna()
test_set.fillna(0)
print(train_set.isnull().sum())
print(test_set.isnull().sum())
#그냥 dropna만 붙이면 삭제가 안되고 재정의해야함
train_set = train_set.dropna()
test_set = test_set.fillna(0)
print(train_set.isnull().sum())
print(test_set.isnull().sum())
#삭제됨 그러나 난값나옴 x,y를 밑으로 안내려서 그럼 데이터에 숫자값이 아닌 오브젝트 문자값이 포함돼있었던게 문제
#트레인 테스트 나누기
print(train_set.head())
x = train_set.drop(['PassengerId', 'Survived', 'Sex', 'Name', 'Ticket', 'Cabin','Embarked'], axis=1)
y = train_set['Survived']
test_set = test_set.drop(['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin','Embarked'], axis=1)
print(x.shape)
print(x.head())
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=133)
#난수지정안하면 한쪽에서 뜯어가 해당특성이 소실될 위험이 있으므로 붙여준다

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'val_loss', mode='mim', patience=20, restore_best_weights=True)
#얼리스탑핑 발로스 최저로스율 웨이트값저장 
model = Sequential()
model.add(Dense(5, input_dim=7))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
#시그모이드냐 소프트맥스냐 서바이브는 죽었냐 살았냐 1,0이다 즉 시그모이드
model.summary()#계산횟수

#컴파일 훈련
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#시그모이듬이므로 바이너리크로스엔트로피 메트릭스 어큐러시
hist = model.fit(x_train, y_train, epochs=100, batch_size=20, validation_split=0.2, callbacks=es)
#그리기 위한 밑그림
#평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print(y_predict)
# [nan] 왜 y_predict가 난값이 나오는가?
#데이터에 난값이 포함, 데이터에서 난값드랍
# [[0.67215645]
#  [0.5849917 ]
#  [0.6325625 ]
#  [0.65553164]
#  [0.62845546]
# 현재 y_predict값 간단하게 식으로 해결해보겠다
y_predict[(y_predict<0.5)] = 0
y_predict[(y_predict>=0.5)] = 1
#특이하게 대괄호 리스트 안에 튜플형식으로 들어가있다 식은 바뀌면 안되어서 그런듯 싶다 재출력
print(y_predict)
#결과값은 형편없지만 일단 진행을 위해 답안지를 작성해보겠다
#accuracyscore
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predict)
print('loss', loss)
print('acc', acc)
#Y_summit값이 곧 y_predict값?
# gender_submission['Survived'] = y_predict
#불러온 파일의 답안지열에 y_predict 즉 y_summit값을 넣고
# gender_submission.to_csv(path +'submission.csv')
# ValueError: Length of values (55) does not match length of index (418)
#아예 입력하지 못한것 같다
# 답안지 행은 418개인데 y_predict는 55개라서 빈값을 채워줘야 한다
#y-summit과 y프레딕트는다르다 y_summit에는 캐글이 제시한 테스트셋이 y_predict에는 캐글의 트레인셋에서 내가 분리한 x_test셋이 들어가기 때문이 행이 다르고 들어가는 데이터도 다른 것이다
print(test_set)
y_summit = model.predict(test_set)
print(y_summit)
print(y_summit.shape)
# ValueError: Failed to convert a NumPy array to a Tensor (Unsupported object type int).
# test-set을 재정의하지 않아서 drop axis=1했음에도 불구하고 열이 날아가지 않았다

#일단입력
# ValueError: Length of values (87) does not match length of index (418)
#y_summit값은 87행 입력해야하는 값은 418행
#=트레인셋을 dropna하는 것은 예측성능에 영향을 줄 뿐이지만 캐글에서 준 test_set을 dropna하면 답안에 필요한 행 자체를 날리는 결과
#fillna로 바꿔준다
# (418, 1)
#그러나 
# 75066]
#  [0.4676407 ]
#  [0.884499  ]
#  [0.4763826 ]
#  [0.4676407 ]
#  [0.44032136]]
# 이런 값이 들어가면 안되므로 처리해줘야함
y_summit [(y_summit <0.5)] = 0  
y_summit [(y_summit >=0.5)] = 1 
print(y_summit)
print(y_summit.shape)
# Y_summit 상태에서 처리
gender_submission['Survived'] = y_summit
# gender_submission.to_csv('submission.csv')    +
# path를 추가 안하니 그냥 폴더에 csv 생성됨
gender_submission = gender_submission.astype(int)
gender_submission.to_csv(path + 'submission.csv', index=False)
#index 자동생성돼서 삭제
#결과값이 1이아니라 1.0이 나옴 .astype(int) 추가해줌

# print(y_summit.isnull().sum())
# AttributeError: 'numpy.ndarray' object has no attribute 'isnull'
# submission = gender_submission.fillna(gender_submission.mean())
# 오류메세지때문에 써야하는것은 맞는데 메커니즘은 모름
