from keras.datasets import imdb
import numpy as np
import pandas as pd
(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=10000
)

print(x_train) #갯수는 나오지만 각 데이터마다의 길이는 알 수 없다
print(x_train.shape, x_test.shape) #(25000,) (25000,)
print(np.unique(y_train, return_counts=True)) #(array([0, 1], dtype=int64), array([12500, 12500], dtype=int64))
# print(len(y_train), len(x_train)) #10105 10105
# print(len(np.unique(y_train))) #46

print(type(x_train)) #'numpy.ndarray
print(type(x_train[0]))
print(len(x_train[0])) #218
print(len(x_train[1])) #218
# #최대 최소 길이를 알기위해
print('뉴스기사 최대길이', max(len(i)for i in x_train)) #2494
print('뉴스기사 평균길이', sum(map(len, x_train))/len(x_train)) #뉴스기사 평균길이 238.71364

# #100개정도로 자른다
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
x_train = pad_sequences(x_train, padding='pre', maxlen=100, truncating='pre')
# #8982, -> 8982,100
x_test = pad_sequences(x_test, padding='pre', maxlen=100, truncating='pre')

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, y_train.shape) #(25000, 100) (25000, 2)
print(x_test.shape, y_test.shape)#(25000, 100) (25000, 2)

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten

model = Sequential()
model.add(Embedding(3,6,input_shape=(100,)))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=10, batch_size=10)

acc = model.evaluate(x_test, y_test)

