import numpy as np
import pandas as pd
from sklearn.datasets import load_boston, load_breast_cancer,fetch_covtype,load_wine,load_iris,load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
print(np.unique(data.target))

kmeans = KMeans(n_clusters=2,random_state=32478)

df = pd.DataFrame(data.data,columns=data.feature_names)
print(df)
df['cluster'] = kmeans.fit(df).labels_
df['target'] = data.target
print(df['target'])
print(df['cluster'])
print(accuracy_score(df['target'],df['cluster']))

# 0.8541300527240774
# 어큐러시는 시리즈나 np나 다 받는다