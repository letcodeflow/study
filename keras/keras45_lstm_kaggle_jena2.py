#split lstm conv1d rnn cnn 시계열 데이터
import pandas as pd
import numpy as np
import time
data = pd.read_csv('c:/study/_data/kaggle_jena/jena_climate_2009_2016.csv')
data = data.transpose()
print(data[:5])
print(data.shape) #(15, 420551)
def split(seq, size): 
    aaa = []
    for i in range(420551 - 574):
        subset = np.reshape(seq.iloc[:,i*size-1:575+size*i],(seq.iloc[:,i*size-1:575+size*i],1))
        aaa.append(subset)
    print(type(aaa))
    return aaa
size = 144
s_time = time.time()
b = split(data,size)
print(b)
print(time.time()-s_time)
print(b.shape)
data2 = data.iloc[:,-1:575]
print(data2)
data3 = data.iloc[:,0:719]
print(data3)
data4 = data.iloc[:,420550:]
print(data4)
# data5 = data.iloc[:,431:575]
# print(data5)
# def mix(a):
#     subset = a.iloc[:,:100]
#     return print(subset.shape)
# b = mix(data)
