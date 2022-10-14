import numpy as np

a = np.array([-10,2,3,4,5,6,700,8,9,10,11,12,50])
a = a.reshape(-1,1)
print(a.shape)
from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=.1) #데이터 범위에서 10프로 이상치
b = a[:,0]
outliers.fit(b)
result = outliers.predict(b)
print(result)