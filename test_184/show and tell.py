import numpy as np
p = np.array(range(48)).reshape(6,8)
print('p',p, p.shape)
a = p.reshape(-1,4,2)
print('a',a, a.shape)
c = np.transpose(a, (1,0,2))
print('c',c, c.shape)