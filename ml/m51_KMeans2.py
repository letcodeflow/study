import numpy as np
import pandas as pd
from sklearn.datasets import load_boston, load_breast_cancer,fetch_covtype,load_wine,load_iris,load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

data = load_wine()
df = pd.DataFrame(data.data,columns=data.feature_names)
print(np.unique(data.target,return_counts=True))
kmeans = KMeans(n_clusters=3,random_state=234)
kmeans.fit(df)

print(data.target)
print(kmeans.labels_)
print(accuracy_score(data.target,kmeans.labels_))