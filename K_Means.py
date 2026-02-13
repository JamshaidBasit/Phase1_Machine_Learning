#K-Means Clustering (Centroid-Based)
#Logic: Ye data ko $K$ groups mein baant deta hai. Har group ka ek center (Centroid) hota hai. Points apne sabse qareebi centroid ke group mein chale jaate hain.
#Interview Tip: "Elbow Method" use hota hai best $K$ dhoondne ke liye.

#Coding Methods:
#n_clusters: Batana parta hai kitne groups banane hain.
#inertia_: Ye batata hai ke points centroids se kitne door hain (jitna kam, utna behtar).
#cluster_centers_: Centroids ki coordinates nikalne ke liye.

import pandas as pd
from sklearn.cluster import KMeans

data=pd.read_csv("house_data.csv")
print(data.head())

X=data[["Area","Bedrooms","Age"]]
kmeans1=KMeans(n_clusters=3,random_state=42)

cluster=kmeans1.fit_predict(X)
print(cluster)


## different value of k 
intertia_value=[]
for k in range(1,11):
    kmeans2=KMeans(n_clusters=k,random_state=42)
    kmeans2.fit(X)
    intertia_value.append(kmeans2.inertia_)
print(intertia_value)


