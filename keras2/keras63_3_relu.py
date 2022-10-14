import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0,x) #0이하의 값은 0 이상의 값은 그대로

relu = lambda x: np.maximum(0,x)

#elu,selu,reaky relu

