
#데이터 불러오기 캐글 캘리포니아가 아니라서 다행이다
from sklearn.datasets import fetch_california_housing
data_set=fetch_california_housing()
print(data_set.feature_names)
print(data_set.DESCR)

x = data_set.data
y = data_set.target
    # :Number of Instances: 20640

    # :Number of Attributes: 8 numeric, predictive attributes and the target
#,xy나누기
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.7, shuffle=False
)
#모델구성
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

model = Sequential()
model.add(Dense(5, input_dim=8))
model.add(Dense(5, activation='sigmoid'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))

#컴파일 훈련
model.compile(loss = 'mae', optimizer='adam',
              metrics=['accuracy', 'mse'])
from tensorflow.python.keras.callbacks import EarlyStopping
ES = EarlyStopping(monitor = 'val_loss', patience=20,
                   mode='min', restore_best_weights=True,
                   verbose=3)
hist = model.fit(x_train, y_train, epochs=100, 
                 validation_split=0.2,
                  callbacks=[ES],
                 verbose=2)
#평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)

print('loss', loss)
print('r2', r2)

#프린트 맨나중 그리기먼저
import matplotlib.pyplot as plt
plt.figure(figsize=(5,2))
plt.plot(hist.history['loss'], marker='.',c='red',label='loss')
plt.plot(hist.history['val_loss'], marker='.',c='blue',label='val_loss')
plt.grid()

from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

plt.xlabel('좋아')
plt.ylabel('모든것이좋아')
plt.legend()
plt.show()
print('loss', loss)
print('r2', r2)

# 미적용
# loss [0.7018911838531494, 0.002099144272506237, 0.8068785071372986]
# r2 0.4429087256199721

# 적용
# loss [1.1692869663238525, 0.002099144272506237, 2.441298723220825]
# r2 -0.6855401595599606

# 노드와 레이어 수가 적어서 그런지 상당히 하락했다