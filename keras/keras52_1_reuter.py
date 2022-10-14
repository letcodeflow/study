from keras.datasets import reuters
import numpy as np
import pandas as pd

(x_train, y_train), (x_test, y_test)=reuters.load_data(
    num_words=10000, test_split=0.2 #단어사전 갯수 10000
)

print(x_train) #갯수는 나오지만 각 데이터마다의 길이는 알 수 없다
print(np.unique(y_train, return_counts=True))
print(len(y_train), len(x_train)) #10105 10105
print(len(np.unique(y_train))) #46

print(type(x_train)) #'numpy.ndarray
print(type(x_train[0]))
print(len(x_train[0])) #87 list타입이기 때문에 shape으로는 알수없다 
#최대 최소 길이를 알기위해
print('뉴스기사 최대길이', max(len(i)for i in x_train)) #2376
# print('뉴스기사 최소길이', min(len(i)for i in x_train))
print('뉴스기사 평균길이', sum(map(len, x_train))/len(x_train)) #145.84948045522017
#100개정도로 자른다
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
x_train = pad_sequences(x_train, padding='pre', maxlen=100, truncating='pre')
#8982, -> 8982,100
x_test = pad_sequences(x_test, padding='pre', maxlen=100, truncating='pre')
print(x_train.shape, y_train.shape) #(8982, 100) (8982, 46)
print(x_test.shape, y_test.shape)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, y_train.shape) #(8982, 100) (8982, 46)
print(x_test.shape, y_test.shape)#(1123, 100) (1123, 46)

#2.모델
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten

model = Sequential()
model.add(Embedding(187,20000, input_shape=(100,)))
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(46, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(x_train,y_train, epochs=1,batch_size=100, validation_split=0.2)

acc = model.evaluate(x_test, y_test)
print(acc)
