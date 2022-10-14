from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input,Conv1D, LSTM,Reshape
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
#1.데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)
print(x_train.shape)
import numpy as np
print(np.unique(y_train,return_counts=True))


from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train[:5])
#acc 0.98이상
#cnn 3개이상

#2.모델구성
# model = Sequential()
# model.add(Conv2D(filters=64, kernel_size=(3,3), padding='valid', input_shape=(28,28,1))) 
# model.add(MaxPooling2D())
# model.add(Conv2D(7, (2,2), padding='valid', activation='relu'))
# model.add(Conv2D(7, (2,2), padding='valid', activation='relu'))
# model.add(Conv2D(7, (2,2), padding='valid', activation='relu'))
# model.add(Flatten())
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(10, activation='softmax')) 
# model.summary()


input1 = Input(shape=(28,28,1))
con1 = Conv2D(32, (16,16), padding='same')(input1)
max = MaxPooling2D()(con1)
con2 = Conv2D(100, (2,2))(max)
res = Reshape((13,1300))(con2)
output1 = Conv1D(6,2,padding='same')(res)
# lstm1 = LSTM(18,2)(con3)
# den1 = Dense(10)(lstm1)
# output1 = Dense(2)(den1)
model = Model(inputs=input1, outputs=output1)
model.summary()
# #3.컴파일 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam')
# model.fit(x_train, y_train, epochs=10, batch_size=20)
#발리 콜백 생략
""" 
print(y_test)
#4.평가 훈련
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print(y_test)
print(y_predict)
y_predict = np.argmax(y_predict, axis= 1)
y_test = np.argmax(y_test, axis= 1)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predict)
print(acc)

# 0.9809
#  """