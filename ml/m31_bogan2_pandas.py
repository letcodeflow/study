import pandas as pd
import numpy as np

data = pd.DataFrame([[2,np.nan,6,8,10],
                     [2,4,np.nan,8,np.nan],
                     [2,4,6,8,10],
                     [np.nan,4,np.nan,8,np.nan]])

data = data.T
data.columns = ['a','b','c','d']
print(data)

#결측치 확인
print(data.isnull())
print(data.info())

#1.삭제
print('==================')
print(data.dropna())
print(data.dropna(axis=1))

#2-1. 특정값 -평균
means = data.mean()
print('means\n',means)
data2 = data.fillna(means)
print(data2)

#2-2. 특정값 -중위
means = data.median()
print('means\n',means)
data3 = data.fillna(means)
print(data3)

#2-3 특정값 ffill bfill
print('======================')
data4 = data.fillna(method='ffill') #앞에 걸로 채우겠다
print(data4)

#2-4 특정값 bfill
data5 = data.fillna(method='bfill')
print(data5)
#2-5 특정값 임의
print('================')
data6 = data.fillna(38746)
print(data6)
###################### 특정칼럼만
means = data['a'].mean()
print(means)
data['a'] = data['a'].fillna(means)
print(data)

meds = data['b'].median()
data['b'] = data['b'].fillna(meds)
print(data)