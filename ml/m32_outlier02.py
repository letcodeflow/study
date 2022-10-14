import pandas as pd
import numpy as np
a = np.array([[-10,2,3,4,5,6,7,8,9,10,11,12,50],
            [100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]])
a =a.T
print(a.shape)
print(a)
# a = pd.DataFrame(a)
def outliers(data_out):
    quartile_1,q2,quartile_3 = np.percentile(data_out[:,0],[25,50,75])
    quartile_11,q21,quartile_31 = np.percentile(data_out[:,1],[25,50,75])
    print('1사분위:',quartile_1)
    print('1사분위:',quartile_11)
    print('q2:',q2)
    print('q2:',q21)
    print('3사분위:',quartile_3)
    print('3사분위:',quartile_31)
    iqr = quartile_3 -quartile_1
    iqr1 = quartile_31 -quartile_11
    print('iqr:',iqr)
    print('iqr:',iqr1)
    lower_bound = quartile_1 -(iqr*1.5)
    lower_bound1 = quartile_11 -(iqr1*1.5)
    upper_bound = quartile_3 +(iqr*1.5)
    upper_bound1 = quartile_31 +(iqr1*1.5)
    print(np.where((data_out[:,0]>upper_bound)|(data_out[:,0]<lower_bound)))
    print(np.where((data_out[:,1]>upper_bound1)|(data_out[:,1]<lower_bound1)))

    
outliers_loc = outliers(a)
print('이상치 위치',outliers_loc)
# import matplotlib.pyplot as plt
# plt.boxplot(a)
# plt.show()