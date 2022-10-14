import pandas as pd 
import numpy as np 
import os 

path = 'C:/Users/aiapalm/OneDrive - KNOU/beat/colab testin/dacon-kaggle-any/chung/'

x = pd.read_csv(path + 'train_input/CASE_01.csv',index_col=0)[:1440]

print(x.shape) #(1440, 37)
# x.to_csv('D:\study_data\_temp/x.csv')
y = pd.read_csv(path + 'train_target/CASE_01.csv',index_col=0)[:1]

print(y)
from xgboost import XGBRegressor
from tensorflow.python.keras.layers import Dense,SimpleRNN,LSTM
from tensorflow.python.keras.models import Sequential
x = x.to_numpy()

model = XGBRegressor()
model.fit(x,y)
print("gg : ",  model.score(x,y))
