import numpy as np
a, b = np.arange(0,8).reshape(2,2,2), np.arange(0,8).reshape(2,2,2)
print(a)
print(b,end='\n')
print(a*b)
print(a@b)
print(np.matmul(a,b))