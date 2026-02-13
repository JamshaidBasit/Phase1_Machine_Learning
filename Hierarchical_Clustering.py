#Hierarchical Clustering
#Logic: Ye ek "Tree" (Dendrogram) banata hai. Ya toh har point ko ek cluster maan kar merge karta jata hai (Agglomerative), ya ek bade cluster ko torta jata hai.
#Advantage: Aapko pehle se nahi batana parta ke kitne clusters chahiye.
#Coding Methods:
#linkage: 'ward', 'complete', 'average' (Merge karne ka tarika).
#scipy.cluster.hierarchy: Dendrogram plot karne ke liye best library hai.

import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

data=pd.read_csv("house_data.csv")
print(data.head())

X=data[["Area","Bedrooms","Age"]]

hc=AgglomerativeClustering(n_clusters=3,linkage='ward')
cluster1=hc.fit_predict(X)

print(cluster1)


sch.dendrogram(sch.linkage(X,method='ward'))