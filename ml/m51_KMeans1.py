from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

datasets = load_iris()
df = pd.DataFrame(datasets.data,columns=datasets.feature_names)
print(df)
kmeans = KMeans(n_clusters=2,random_state=32)
kmeans.fit(df)

print(kmeans.labels_)
print(datasets.target)

print(accuracy_score(datasets.target,kmeans.labels_))

df['cluster']  = kmeans.labels_
df['target'] = datasets.target