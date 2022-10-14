import numpy as np
import matplotlib.pyplot as plt

f = lambda x : x**2 - 4*x+6

x = np.linspace(-1,6,100)
print(x,len(x))

y = f(x)

plt.plot(x,y)
plt.show()
