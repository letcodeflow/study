import numpy as np
from sklearn.decomposition import PCA
from keras.datasets import mnist



(x_train,_),(x_test,_) = mnist.load_data()
print(x_train.shape)

x = np.append(x_train, x_test, axis=0)
print(x.shape)
x = x.reshape(-1,x.shape[1]*x.shape[2])
pca = PCA(n_components=486)
x = pca.fit_transform(x)
print(pca.explained_variance_ratio_)
print(np.cumsum(pca.explained_variance_ratio_))

# import matplotlib.pyplot as plt
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.grid()
# plt.show()

# 0.95 n_com[pnemt 155]
# 0.99 333
# 0.999 486
# 1 706

print(np.argmax((np.cumsum(pca.explained_variance_ratio_)>=0.95+1)))
