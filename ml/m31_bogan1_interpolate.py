#결측치 처리
#1.행 또는 열 삭제
#2. 임의값 
    # 평균 mean
    # 중위 median
    # 0 fillna
    # 앞 ffill
    # 뒤 bfill
    # 특정값 등등
#3.보간 interpolate = 선형회귀
#4.결측치없는모델 predict(결측치)
#5.부스팅계열 - 결측치 이상치에 자유로움

import pandas as pd
import numpy as np
from datetime import datetime

dates = ['8/10/2022','8/11/2022','8/12/2022','8/13/2022','8/14/2022']

dates = pd.to_datetime(dates)
print(dates)
# 시리즈 = 벡터
print('====================')
ts = pd.Series([2,np.nan,np.nan,8,10],index=dates)
print(ts)
print('====================')
ts = ts.interpolate()
print(ts)