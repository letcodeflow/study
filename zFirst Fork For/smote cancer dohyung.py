from dataclasses import replace
import pandas as pd 
import numpy as np 
from sklearn.datasets import load_breast_cancer
# 1.데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)
x = np.delete(x,np.where(y==0)[0][:112],0)
y = np.delete(y,np.where(y==0)[0][:112],0)
print(x.shape, y.shape)
print(np.where(y==0)[0][:112].shape)
print(type(np.where(y==0)[0]))
# zero = np.array(np.where(y==0))
# zero2 = zero[:,:112]
# print(type(np.array(np.where(y==0))))
# print(np.where(y==0).shape)
# print(np.array(np.where(y==0))[:112])

# zero = np.array(np.where(y==0))
# zero2 = zero.reshape(212,)[:112]
# zero3= np.random.choice(zero2,112,replace=False) # 섞고싶으면 !
# x = np.delete(x,zero2,0)
# print(x.shape, y.shape)