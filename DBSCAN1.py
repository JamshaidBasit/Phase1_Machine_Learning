#DBSCAN (Density-Based)
#Logic: Ye "bheer" (density) dekhta hai. Jahan data points qareeb-qareeb hain unhe ek cluster bana deta hai, aur jo points akele hain unhe Noise/Outliers declare kar deta hai.
#Speciality: Ye ajeeb shapes (circles, moons) ke clusters bhi dhoond leta hai jo K-Means nahi kar sakta.

#Coding Methods:
#eps (Epsilon): Do points ke darmiyan max fasla.
#min_samples: Ek cluster banane ke liye kam az kam kitne points chahiye.
#labels_: Resulting clusters (-1 ka matlab hai Outlier/Noise).

import pandas as pd
from sklearn.cluster import DBSCAN

data=pd.read_csv("house_data.csv")
print(data.head())

X=data[["Area","Bedrooms","Age"]]

dbscan1=DBSCAN(eps=0.5,min_samples=2)
cluster1=dbscan1.fit_predict(X)
print(cluster1)

outlier=(cluster1==-1).sum()
print(outlier)