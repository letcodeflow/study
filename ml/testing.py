from random import random
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

# print(dir(datasets))
p = x.__iter__()
print(p.__next__())
# print(dir(x))
print(x)
......
..