import numpy as np
from torch import gradient

f = lambda x : x**2 - 4*x+6

def f(x):
    temp = x**2 - 4*x+6
    return temp

gradient = lambda x: 2*x -4

x = -100000.0
epochs = 20
learning_rate = 0.25

print('{:02d}\t {:6.5f}\t {:6.5f}\t'.format(0,x,f(x)))

for i in range(epochs):
    x = x - learning_rate*gradient(x)
    print('{:02d}\t {:6.5f}\t {:6.5f}\t'.format(i+1,x,f(x)))

