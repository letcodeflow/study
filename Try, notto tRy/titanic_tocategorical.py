

from subprocess import call
import pandas as pd


path = './_data/kaggle_titanic/'
train_set = pd.read_csv(path + 'train.csv', index_col=0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)


# print(train_set['Survived'].unique())
# #서바이브드 고유라벨값이 무엇인지
# print(train_set['Survived'].value_counts())
# #value counts쓰면 한번에 나오는듯? 라벨값이 뭐고 몇개인지
# print(train_set['Embarked'].value_counts())

from sklearn.preprocessing import LabelEncoder
#sk런에서 라벨엔코더를 임포트한다
encoder = LabelEncoder()
#엔코더를 불러와서 이름을 정해준다
encoder.fit(train_set['Sex']) 
#정의된 엔코더로 트레인셋 sex의 고유값을 분류한다 = 학습한다

print(encoder.classes_)
#['female' 'male']
for i, label in enumerate(encoder.classes_):
    print(i, '->', label)
#긁어온것 분류한 고유값과 지정된 값을 출력하도록 한다

tatu1 = encoder.transform(train_set['Sex'])
#이를 변환하고 다른 이름으로 저장한다
#inverse_trainsform 하면 숫자값을 고유값으로 변환할 수 있다
print(encoder.inverse_transform([0,1]))
#['female' 'male']

encoder.fit(train_set['Embarked'])
print(encoder.classes_)

for i, label in enumerate(encoder.classes_):
    print(i, '->', label)
tatu2 = encoder.transform(train_set['Embarked'])
#엠바크드 마찬가지
print(encoder.inverse_transform([0,1,2]))
#['C' 'Q' 'S'] 숫자를 고유라벨로 변환해 출력한다
#해야할것 해당 인수들을 기존 데이터셋에 넣어야한다
#해당열을 재정의하는 방식으로 계산된걸 다시 넣을 수 있다
train_set['Sex'] = encoder.fit_transform(train_set['Sex'])
train_set['Embarked'] = encoder.fit_transform(train_set['Embarked'])
#인수가 리스트형식으로 들어가지는 않는듯 싶다
#sex와 embarked에 숫자형식으로 들어간거 확인
print(train_set['Sex'])


import numpy as np
#이제 원핫인코딩
#불러오기
from sklearn.preprocessing import OneHotEncoder
#정의하기
ohe = OneHotEncoder(sparse=False)
#sparse=True가 디폴트이며 이는 Matrix를 반환한다.
# 원핫인코딩에서 필요한 것은 array이므로 sparse 옵션에 False를 넣어준다.
#사용하기
train_set['Sex'] = ohe.fit_transform(train_set['Sex'].values.reshape(-1,1))
print(train_set['Sex'])
print(train_set['Sex'].shape)

# train_set['Sex'].toarray()
train_set['Embarked'] = ohe.fit_transform(train_set['Embarked'].values.reshape(-1,1))
print(train_set['Embarked'])
print(train_set['Embarked'].shape)

# print(type(ohe.categories_))
# #핫인코더가 나눈 카테고리는 리스트형태이다 이를 풀어야한다
# print(ohe.categories_[0])
#sex열 0번째행의 인수를 확인한다





print(train_set.head())

# 만약 fit호출과정에서 보지못했던 클래스가 transform호출 시 나타나면 아래 오류 메시지가 발생한다.
# "ValueError: unknown categorical feature present [  ] during transform"
# 인코딩할 때는 sklearn OneHotEncoder를 이용한다.
# 반대로 Decoder는 제공하지 않고 있어 디코딩할 때는 numpy의 argmax를 이용한다.

# train_set = pd.get_dummies(train_set, columns = ['Sex', 'Embarked'])
# print(list(train_set.columns))
# #트레인셋

print(train_set.head())
# print(train_set.columns.values)

#섹스랑 embarked tocategorical로 원핫인코딩하기 그러려면 라벨인코딩으으로 고유숫자값으로 변환해야함

# y=np.array([0,1,0,1,2])
# print(y)
# print(y.shape)
# y=np.array([0,1,0,1,2]).reshape(-1,1)
# print(y)
# print(y.shape)




train_set=train_set.dropna()
test_set=test_set.dropna()



#일단 떨군다


#        PassengerId    Survived      Pclass         Age       SibSp       Parch        Fare
# count   891.000000  891.000000  891.000000  714.000000  891.000000  891.000000  891.000000
# mean    446.000000    0.383838    2.308642   29.699118    0.523008    0.381594   32.204208
# std     257.353842    0.486592    0.836071   14.526497    1.102743    0.806057   49.693429
# min       1.000000    0.000000    1.000000    0.420000    0.000000    0.000000    0.000000
# 25%     223.500000    0.000000    2.000000   20.125000    0.000000    0.000000    7.910400
# 50%     446.000000    0.000000    3.000000   28.000000    0.000000    0.000000   14.454200
# 75%     668.500000    1.000000    3.000000   38.000000    1.000000    0.000000   31.000000
# max     891.000000    1.000000    3.000000   80.000000    8.000000    6.000000  512.329200
#        PassengerId      Pclass         Age       SibSp       Parch        Fare
# count   418.000000  418.000000  332.000000  418.000000  418.000000  417.000000
# mean   1100.500000    2.265550   30.272590    0.447368    0.392344   35.627188
# std     120.810458    0.841838   14.181209    0.896760    0.981429   55.907576
# min     892.000000    1.000000    0.170000    0.000000    0.000000    0.000000
# 25%     996.250000    1.000000   21.000000    0.000000    0.000000    7.895800
# 50%    1100.500000    3.000000   27.000000    0.000000    0.000000   14.454200
# 75%    1204.750000    3.000000   39.000000    1.000000    0.000000   31.500000
# max    1309.000000    3.000000   76.000000    8.000000    9.000000  512.329200


#패신저 아이디 삭제

#xy 설정하고 테스트 트레인셋 나눈고 모델구성 


from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split

x = train_set.drop(['Survived','Name', 'Ticket', 'Cabin'], axis=1)
y = train_set['Survived']
print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=137)

model = Sequential()
model.add(Dense(50, input_dim=7))
# print(x.shape) #5
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#컴파일 훈련
from tensorflow.python.keras.callbacks import EarlyStopping
ES = EarlyStopping(monitor='val_loss', mode='min', patience=50, restore_best_weights=True)

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
hist = model.fit(x_train, y_train, epochs = 100, batch_size=32, validation_split=0.2, callbacks=ES)

#평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

print(y_test)
print(y_predict)
print(y_test.shape)

import numpy as np
y_predict = np.argmax(y_predict, axis= 1)

#패신저 아이디가 왜 y_test에 붙어서 나오는가?


from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predict)

print('loss', loss)
print('acc', acc)

#현재 문제 y_predict값 nan값 나옴 이해못함 아마도 라벨인코딩 argmax 조합인것 같은데 다시 봐야함
#pandas 함수 적용못함

# y_summit = model.predict(test_set)
# submission = pd.read_csv(path + 'gender_submission.csv')
# submission['Survive'] = y_summit
# submission.to_csv(path + 'submission.csv')
# ValueError: Failed to convert a NumPy array to a Tensor (Unsupported object type int).

#현재 라벨인코딩 밋 원핫인코딩 3개 쓸수 없음으로 인해서 데이터셋 na를 전부 날렸다 알게되면 원핫인코딩으로 숫자변환해 쓸수 있을 것 추후 전처리를 배우면 상관관계를 분석해 이용할 수?
#argmax 의 의의를 제대로 이해못함
#아침에 와서 모든 기술이 다 들어간 코드 작성해보기

