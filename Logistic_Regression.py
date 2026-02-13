#Logic: Naam regression hai lekin ye Classification ke liye use hota hai. Ye "Sigmoid Function" use karke output ko 0 aur 1 ke beech rakhta hai (Probability).
#Use Case: Spam detection, Disease prediction.
#Coding Methods:
#predict_proba(X): Ye batata hai ke 0 hone ka kitna chance hai aur 1 hone ka kitna.
#decision_function(X): Raw scores nikalne ke liye.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


data=pd.read_csv("house_data.csv")
print(data.head())

data["Expensive"] = (data["Price"] > 300000).astype(int)

x=data[["Area","Bedrooms","Age"]]
y=data["Expensive"]
print(y)

X_train,x_test,Y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=42)

logr=LogisticRegression(max_iter=1000)

logr.fit(X_train,Y_train)

print(logr.predict_proba(x_test))
print(logr.decision_function(x_test))




