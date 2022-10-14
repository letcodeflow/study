from keras.datasets import imdb
from keras_preprocessing.text import Tokenizer
import numpy as np
(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=10000, 
)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)
print(np.unique(y_train, return_counts=True))

print(type(x_train[0]))
print((x_train[0]))
print(len(x_train[1]))
print(len(x_train[2]))
print(len(x_train[3]))

print(len(x_train))
print('리뷰의 최대길이', max(len(i) for i in x_train))
print('리뷰의 평균 길이', sum(len(i)for i in x_train)/len(x_train))

from keras_preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x_train, padding='pre', maxlen=200, truncating='pre')
x_test = pad_sequences(x_test, padding='pre', maxlen=200, truncating='pre')
#y고유값은 2개이므로 넘어간다

print(pad_x.shape, x_test.shape, y_train.shape, y_test.shape)
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Conv1D, Conv2D, LSTM, GRU, Embedding

in1 = Input(shape=(200,))
em1 = Embedding(input_dim=5, output_dim=10, input_length=10)(in1)
L1  = LSTM(10)(em1)
D1  = Dense(1)(L1)
model = Model(inputs = in1, outputs = D1)
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(pad_x, y_train, epochs=10, batch_size=10, )

loss = model.evaluate(x_test, y_test)
print(y_test)
pred = model.predict(x_test)
print(pred.round().astype(int))