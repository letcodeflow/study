import pandas as pd
import numpy as np
sam = pd.read_csv('c:/study/_data/test_amore_0718/삼성전자220718.csv',encoding='cp949',thousands=',')
amore = pd.read_csv('c:/study/_data/test_amore_0718/아모레220718.csv',encoding='cp949',thousands=',')
# print(sam.columns)
print(sam.shape)
print(amore.shape)

sma = sam.dropna()
amore = amore.dropna()
print(sam.shape)
print(amore.shape)

sam = sam[1:1035]
amore = amore[1:1035]
print(sam.shape)
print(amore.shape)

sam.rename(columns={'일자':'day'}, inplace=True)
sam = sam.sort_values('day',ascending=True)
amore.rename(columns={'일자':'day'}, inplace=True)
amore = amore.sort_values('day',ascending=True)
samx = sam[['고가', '저가', '등락률', '거래량',
       '금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비']]
samy = sam[['시가', '종가']]
amorex = amore[['고가', '저가', '등락률', '거래량',
       '금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비']]
amorey = amore[['시가', '종가']]
print(amorey.shape)
print(sam)
# print(sam)
print(amorey.isnull().sum())
def split_x(seq, size): #함수 split_x는 한 
    aaa = []
    for i in range(len(seq) - 24):
        subset = seq[i:(i+size)]
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)
size=20
samx = split_x(samx,size)
amorex = split_x(amorex,size)
# amorex = split_x(amorex[],size)

size=3
def split_y(seq, size): #함수 split_x는 한 
    aaa = []
    for i in range(len(seq) - 23):
        subset = seq[i+20:(i+20+size)]
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)
samy = split_y(samy,size)
amorey = split_y(amorey,size)
print(amorex.shape)
print(amorey.shape)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
scaler = MinMaxScaler()
# scaler = MaxAbsScaler()

samx = samx.reshape(20200,12)
samx = scaler.fit_transform(samx)
# x1_test = x1_test.reshape(204*20,7)
# x1_test = scaler.transform(x1_test)

amorex = amorex.reshape(20200,12)
amorex = scaler.fit_transform(amorex)
# x2_test = x2_test.reshape(204*20,7)
# x2_test = scaler.transform(x2_test)

# Conv1D에 넣기 위해 3차원화
samx = samx.reshape(1010, 20, 12)
# x1_test = x1_test.reshape(204, 20, 7)
amorex = amorex.reshape(1010, 20, 12)
# x2_test = x2_test.reshape(204, 20, 7)
# samx = samx[1:953]
# samy = samy[1:953]
# amorex = amorex[1:953]
# amorey = amorey[1:953]
print(samx.shape, amorex.shape, samy.shape, amorey.shape)
#3.모델구성
from keras.models import Model
from keras.layers import Input, LSTM, Dense, concatenate,Conv1D,Reshape,GRU

in1 = Input(shape=(20,12))
lstm1 = LSTM(3,name='l1', activation='sigmoid')(in1)
dens1 = Dense(10,name='d1')(lstm1)
out1 = Dense(10,name='d13')(dens1)

in2 = Input(shape=(20,12))
lstm2 = GRU(3,name='l2')(in2)
dens2 = Dense(10,name='d2')(lstm2)
out2 = Dense(10,name='d23')(dens2)

in3 = concatenate((out1,out2))
dens3 = Dense(13)(in3)
dens32 = Dense(80, activation='relu')(dens3)
resha1 = Reshape((4,20))(dens32)
out3 = Conv1D(2,2,name='d3')(resha1)

model = Model(inputs=[in1,in2], outputs = out3)
model.summary()
hist = model.load_weights('c:/study/_test/ddserenade.h5')

model.compile(loss = 'mse', optimizer='adam')
# hist = model.fit([samx,amorex],amorey, epochs=150, batch_size=100, validation_split=0.2)
# model.save_weights('c:/study/_test/ddserenade.h5')
pred = model.predict([samx,amorex])
print(pred.shape)
print('수요일 종가', pred[1009:1010,2:3,1:2])
# print(pred[])
