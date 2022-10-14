#split lstm conv1d rnn cnn 시계열 데이터
import pandas as pd
import numpy as np
data = pd.read_csv('c:/study/_data/kaggle_jena/jena_climate_2009_2016.csv')
print(data.shape) #(420551, 15) #하루 143행
def split(seq, size): #함수 split_x는 한 
    aaa = []
    for i in range(2900):
        subset = seq.loc[i*size:(i*size)+575]
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)


# print(len(data.loc[:]))
size = 144
gen2 = split(data,size)
print(gen2.shape)
print(gen2)

gen = gen2.T


print(gen.shape)
gen = gen[1:]
x = gen[:,:size*3]
y = gen[:,size*3+1:size*4]
x = np.asarray(x).astype('long')
y = np.asarray(y).astype('double')
print(x.shape, y.shape)
print(x)

# print(x)
# print(x.shape)
from keras.layers import Input, Dense, Conv1D, LSTM, Reshape, Flatten
from keras.models import Model

in1 = Input(shape=(432,2900))
l1 = LSTM(29,return_sequences=True)(in1)
c1 = Conv1D(2900, 290, name = 'c1')(l1)
# r1 = Reshape((1,2900))(l1)
# f1 = Flatten()(l1)
# d1 = Dense(143)(f1)
model = Model(inputs = in1, outputs = c1)
model.summary()

model.compile(loss = 'mse', optimizer='adam')
model.fit(x,y, epochs=1, batch_size=100)
loss = model.evaluate(x,y)
print(loss)
# print(data)
# print(data.shape)
# print(np.unique(data, return_counts=True))
# print(data.value_counts())
# print(data.describe())
# print(data.isnull().sum())
# print(data.info())
# print(data[:5])
# print(data.shape)
# print(data_split.colu)
# data.reshape(6,4205510)
# x=data[:,:-1]
# y=data[:-1]
# print(x.shape, y.shape)
# print(x[:5],y[:5])
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=9812)

# from keras.models import Model
# from keras.layers import Conv2D,LSTM,Flatten,Reshape,Dense,Input
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# input1 = Input(shape=())
# c1 = Conv2D(10,(3,5))(input1)
# r1 = Reshape(4,60)(c1)
# l1 = LSTM(400, activation='relu',name='l1')(r1)
# d1 = Dense(20)(l1)
# out1 = Dense(1)(d1)
# model = Model(inputs=input1, outputs=out1)
# model.summary()

# model.compile(loss='mse', optimizer='adam',metrics='mad')
# model.fit(x,y,epochs=10, batch_size=2000)
# loss = model.evaluate(x_test,y_test)
# print(loss)


# def uni_data(dataset, start_index, end_index, history_size, target_size);
# 	data = []
# 	labels = []

# 	start_index = start_index + history_size
# 	if end_index is None:
# 		end_index = len(dataset) - target_size

# 	for i in range(start_index, end_index):
# 		indices = range(i-history_size, i)
# 		data.append(np.reshape(dataset[indices], (history_size, 1)))
# 		labels. append(dataset[i+target_size])
# 	return np.array(data), np.array(labels)

# data2 = uni_data(data,0,,10,1)f