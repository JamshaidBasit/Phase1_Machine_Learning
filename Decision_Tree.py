#Logic: Ye data ko "Tree" ki tarah split karta hai (e.g., Is Salary > 50k? -> Yes/No). Ye "Entropy" ya "Gini Impurity" use karke best split dhoondta hai.
#Use Case: Credit scoring, Loan approval.
#Coding Methods:
#get_depth(): Tree kitna lamba bana hai.
#feature_importances_: Batata hai kaunsa column sabse zyada important hai.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data=pd.read_csv("house_data.csv")
print(data.head())

data["Expensive"]=(data["Price"]>300000).astype(int)

x=data[["Area","Bedrooms","Age"]]
y=data["Expensive"]

X_train,x_test,Y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

dtree=DecisionTreeClassifier(max_depth=5, criterion='gini')
dtree.fit(X_train,Y_train)

print(dtree.predict(x_test))

for name, imp in zip(x, dtree.feature_importances_):
    print(f"{name}: {imp}")


print(dtree.score(X_train,Y_train))
print(dtree.score(x_test,y_test))


