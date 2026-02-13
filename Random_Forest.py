#Random Forest (Ensemble Learning)
#Logic: Ye "Wisdom of the Crowd" par chalta hai. Bohat saare Decision Trees banata hai aur unke results ka Average (Regression) ya Majority Vote (Classification) leta hai.
#Use Case: Jahan accuracy bohot zaroori ho.
#Coding Methods:
#n_estimators: Kitne trees banane hain (Default=100).
#oob_score_: Bagging check karne ke liye (Out of Bag score).

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data=pd.read_csv("house_data.csv")
print(data.head())

x=data[["Area","Bedrooms","Age"]]
y=data["Price"]

X_train,x_test,Y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=42)

rf=RandomForestClassifier(n_estimators=100,criterion='gini')

rf.fit(X_train,Y_train)

print(rf.predict(x_test))

Train_Score=rf.score(X_train,Y_train)
print(Train_Score)

Test_Score=rf.score(x_test,y_test)
print(Test_Score)

results=pd.DataFrame({
    "Actual Value:": y_test,
    "Predicted:": rf.predict(x_test) 
})


print(results)