import numpy as np
import catboost
a = np.random.randint(0,100,(3,3,3))
b = np.random.randint(0,10,(3,2,2))
print('bf a',a)
a = a.reshape(-1,3*3)
print('at a',a)

print('bf b',b)
b = b.reshape(-1,2*2)
b = b.reshape(-1,2,2)
print('at b',b)

p = catboost.CatBoostRegressor(loss_function='MultiRMSE', verbose=0)
p.fit(a,b)
print(p.predict(a))
