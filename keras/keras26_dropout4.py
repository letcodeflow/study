import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split

#1.데이터
data_sets = load_breast_cancer()
print(data_sets.DESCR)
# :Number of Instances: 569

#     :Number of Attributes: 30 numeric
#와꾸 확인
print(data_sets.feature_names)

x = data_sets['data']
y = data_sets['target']
# x = data_sets.data
# y = data_sets.target
#sklearn 에서 쓰는 예제불러오기

print(x.shape, y.shape)
print(y)
#y는 0아니면 1이다

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)
from sklearn.preprocessing import MinMaxScaler, StandardScaler #대문자 클래스 약어도 ㄷ대문자로

# scaler = MinMaxScaler()
scaler = StandardScaler()
#스케일러정의
scaler.fit(x_train)
#핏한다
x_train = scaler.transform(x_train)
#컬럼별로 스케일 해야하는데 이것을 함수정의자들이 예상하지 않았을까?
#수치로 바꾼것을 본래 데이터셋으로 재정의한다
x_test = scaler.transform(x_test)
#x_test를 트레인에 맞춰서 변환한 값으로 변환한다
#이부분이 이해가 안됨
print(np.min(x_train))
print(np.max(x_train))
# 0.0
# 1.0
print(np.min(x_test))
print(np.max(x_test))

#2.모델
model=Sequential()
model.add(Dense(180, activation='linear', input_dim=30))
model.add(Dense(100, activation='sigmoid'))
model.add(Dropout(0.3))
model.add(Dense(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(100))
model.add(Dropout(0.3))
model.add(Dense(100, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일 훈련
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy', 
                                                                       'mse'] )
from tensorflow.python.keras.callbacks import EarlyStopping
ES = EarlyStopping(monitor='val_loss', patience=90, mode='min', verbose=1, restore_best_weights=True)
g = model.fit(x_train, y_train, epochs=180, batch_size=10, 
            #   validation_split=0.1, 
              callbacks=[ES], verbose=2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss', loss)

y_predict = model.predict(x_test)
print(type(y_predict))
print(type(y_test))

# print(y_test)
# print(y_predict)

pre2 = y_predict.flatten() # 차원 펴주기
pre3 = np.where(pre2 > 0.4, 1 , 0) #0.5보다크면 1, 작으면 0


from sklearn.metrics import r2_score, accuracy_score
r2=r2_score(y_test, pre3)
acc = accuracy_score(y_test, pre3)
print('loss', loss)
print('r2', r2)
print('acc', acc)
'''
print(y_predict)
import matplotlib.pyplot as plt

from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

plt.figure(figsize=(9,6)) #칸만들기
plt.plot(g.history['loss'], marker=',', c='red', label='loss')
#그림그릴거야
plt.plot(g.history['val_loss'], marker=',', c='blue', label='val_loss')
plt.grid()
plt.title('우결')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right') #라벨값위치 생략시 빈자리에 생성
plt.show()

print(y_predict)
'''
# #vali 적용
# loss [0.6554861664772034, 0.640350878238678, 0.23137257993221283]
# r2 -0.561643835616439
# acc 0.6403508771929824

#미적용
# loss [0.6539859175682068, 0.640350878238678, 0.23066139221191406]
# r2 -0.561643835616439
# acc 0.6403508771929824

#발리데이션값이 적어서 그런지 큰차이 없다


#민맥스
# loss [0.10680505633354187, 0.9736841917037964, 0.019497809931635857]
# r2 0.885733377881724
# acc 0.9736842105263158

#스탠
# loss [0.1440308839082718, 0.9736841917037964, 0.026008574292063713]
# r2 0.885733377881724
# acc 0.9736842105263158

# dropout
# loss [0.08373439311981201, 0.9824561476707458, 0.01696014776825905]
# r2 0.9238222519211493
# acc 0.9824561403508771