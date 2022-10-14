import pandas as pd
import numpy as np
a = np.array([[-10,2,3,4,5,6,7,8,9,10,11,12,50],
            [100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]])
a = a[1]
from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=.1) #데이터 범위에서 10프로 이상치
print(a.shape)
a = a.reshape(-1,1)

print(a)
print(type(a))
outliers.fit(a)
result = outliers.predict(a)
print(result)