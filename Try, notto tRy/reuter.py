from keras.datasets import reuters
from keras_preprocessing.text import Tokenizer

(x_train, y_train), (x_test, y_test) = reuters.load_data()

print(x_train)
print(len(x_train[0]))
print(len(x_train[1]))
#개별 길이는 알수 있음
print(max(len(i) for i in x_train))
print(sum(map(len, x_train))/len(x_train))
#최대
#평균
from keras_preprocessing.sequence import pad_sequences
x_train = pad_sequences(x_train, maxlen=140, truncating='pre', padding='pre')
x_test = pad_sequences(x_test, maxlen=140, truncating='pre', padding='pre')
import numpy as np
print(type(y_train))
print(np.unique(y_train, return_counts=True))
print(y_train[0])
print(y_train[1])
print(len(np.unique(y_train))) #y고유값은 46개임 개별 고유값 하나씩
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
from keras.models import Model
from keras.layers import Input, Dense, LSTM,Embedding,Flatten, GRU

in1 = Input(shape=(140,))
em1 = Embedding(input_dim=10, input_length=10, output_dim=10)(in1)
GR1 = GRU(22, activation='sigmoid')(em1)
D1 = Dense(200, activation='relu')(GR1)
Dl  = Dense(46, activation='softmax')(D1)
model = Model(inputs = in1, outputs=Dl)
model.summary()

# model.load_weights('d:/study_data/_temp/reuter_weights.h5')

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=108, batch_size=10, validation_split=0.1)
model.save_weights('d:/study_data/_temp/reuter_weights.h5')

loss = model.evaluate(x_test, y_test)
pred = model.predict(x_test)
print(pred.shape)
pred = np.argmax(pred, axis=1)
print(pred)
from sklearn.metrics import accuracy_score as acc
ac = acc(pred, y_test)
print(ac)

import matplotlib.pyplot as plt
plt.figure(figsize=(4,6))
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.show()