#어떻게 썼는지는 중요하지만 왜 썼는지도 알아야한다
import tensorflow as tf
tf.random.set_seed(137)
#텐서플로 컴퓨터가 처음 w값을 넣을때 난수값을 지정한다
import numpy as np
#어디쓰지? 넘파이?
#fetch-sovtype 불러와서 완성, 그리기까지
#불러오기
from sklearn.datasets import fetch_covtype
#자 불러왔어 이름 설정해야지?
datasets = fetch_covtype()
#어떤 타입의 데이터가 있나, 몇개의 컬럼인가? 일단 sklearn의 내부 명령어를 사용해보자
# print(datasets.shape) 데이터셋 자체에 대한 shape은 안먹힌다
print(datasets.feature_names)
#['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area_0', 'Wilderness_Area_1', 'Wilderness_Area_2', 'Wilderness_Area_3', 'Soil_Type_0', 'Soil_Type_1', 'Soil_Type_2', 'Soil_Type_3', 'Soil_Type_4', 'Soil_Type_5', 'Soil_Type_6', 'Soil_Type_7', 'Soil_Type_8', 'Soil_Type_9', 'Soil_Type_10', 'Soil_Type_11', 'Soil_Type_12', 'Soil_Type_13', 'Soil_Type_14', 'Soil_Type_15', 'Soil_Type_16', 'Soil_Type_17', 'Soil_Type_18', 'Soil_Type_19', 'Soil_Type_20', 'Soil_Type_21', 'Soil_Type_22', 'Soil_Type_23', 'Soil_Type_24', 'Soil_Type_25', 'Soil_Type_26', 'Soil_Type_27', 'Soil_Type_28', 'Soil_Type_29', 'Soil_Type_30', 'Soil_Type_31', 'Soil_Type_32', 'Soil_Type_33', 'Soil_Type_34', 'Soil_Type_35', 'Soil_Type_36', 'Soil_Type_37', 'Soil_Type_38', 'Soil_Type_39']
print(datasets.DESCR)
#  Classes                        7
#     Samples total             581012
#     Dimensionality                54
#     Features                     int
#xy설정이 안됨
x = datasets.data
y = datasets.target
print(y)
y = datasets.target.reshape(-1,1)
print(y)

#일단 설정하고 프린트로 확인한다 reshape는 sklean 원핫인코딩을 위해 미리 설정해둔다
print(x.shape, y.shape) #(581012, 54) (581012, 1)
print(np.unique(y, return_counts=True))
#넘파이 유니크함수로 y값의 종류와 갯수를 나타낸다
#(array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],
    #   dtype=int64))
    
#x,y 설정했으니 트레인 테스트셋 설정을 해준다
#원핫 엔코딩을 해야하는데 아직 정확히 이해를 못하고 있다
#일단 복붙
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
encoder.fit(y)
y = encoder.transform(y).toarray()
#y를 설정할때  reshape 처리했으므로 np관련 처리는 필요없는 듯 하다
#위치를 train test 나눈 후에 놓아 fit에서 에러가 뜬다

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, random_state=88)

#모델구성
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
ES = EarlyStopping(monitor = 'val_loss', mode ='min', patience=20, restore_best_weights=True, verbose=1)


model = Sequential()
model.add(Dense(60, input_dim=54))
#x의 열이 54니까 dim =54 처음나오는 아웃풋은 그보다 크게한다
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(100, activation='relu'))
model.add(Dense(7, activation='softmax'))
#나오는 값은 7개여야 한다. softmax
#다중분류에서 각 구분값이 0,1,2 일때 이는 수학적인 값이 아니라 구분하기 위한 구분값이다 더하거나 빼면 안된다
#softmax는 제일 큰값을 남기고 나머지는 0으로 처리한다(예상)

#3.컴파일 훈련
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics='accuracy')
#다중분류모델에서 마지막 레이어 softmax와 로스식 categorical crossentropy는 거의 99퍼센트다, 최적화 아직 생략, metrics에 로스식 추가시 리스트형태이므로 ㄷ=대괄호[]를 씌어준다
hist = model.fit(x_train, y_train, epochs=10, verbose=1, callbacks=ES, validation_split=0.2)
#아직은 발리데이션을 딱히 지정한다기 보단 랜덤으로 0.2떼어내는 방식을 취한다

#4.평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
#y_predict = model.predict(x_test[:5])의 형태로 나오는 갯수를 조절할 수 있다
#로스율 및 예측값 나옴 accuracy-score 구하기


# from sklearn.metrics import accuracy_score
# acc = accuracy_score(y_test, y_predict)

#구하기전 y_test, y_predict갑을 출력해서 비교해본다
print(y_test)
print(y_predict)
#test값과 predic값의 공통점은 각행의 값을 합하면 1 이 된다는 것과 가장큰 값이 있다는 것
#그러므로 열을 기준으로 각 행의 값중 제일 큰값을 남겨서 y라벨값이 1~7만 남기고 나머지는 버려서 행렬 (n,)으로 둘다 바꾸어줘야 accuracy_score를 구할수 있다

y_predict = np.argmax(y_predict, axis=1)
#넘파이 argmax 인수값은 필요한 행렬 그리고 axis에는 1일 경우 열기준, 0일경우 행기준 남기는 것으로 생각됨
y_test = np.argmax(y_test, axis=1)
#argmax는 현재 파일에서 7개의 열중 가장 큰값의 인덱스를 반환하는 함수인것같다

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predict)
print('loss', loss)
print('acc', acc)

# loss [1.2064911127090454, 0.48764801025390625]
# acc 0.4876480172572058
# 돌긴돌지만 원핫엔코더가 어떤 방식으로 작동하는지 왜 그런 함수가 쓰이는지는 이해하지 못한 것 같다

import matplotlib.pyplot as plt
plt.gray()
plt.matshow(datasets.images[1])
plt.show()
