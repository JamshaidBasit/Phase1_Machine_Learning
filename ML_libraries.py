#2. XGBoost (Extreme Gradient Boosting)
#Concept: Ye "Gradient Boosting" ka ek advance version hai. Ye ek ke baad ek trees banata hai aur har naya tree purane tree ki ghaltiyon (residuals) ko theek karta hai.
#Speciality: Ye parallel processing karta hai aur missing values ko khud handle kar leta hai.
#Coding Methods:
#n_estimators: Kitne trees banane hain.
#learning_rate: Har tree ki ghalti theek karne ki raftaar (0.01 se 0.3).
#early_stopping_rounds: Agar accuracy behtar hona ruk jaye toh training rok do.

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

data=pd.read_csv("house_data.csv")
print(data.head())

data["Expensive"]=(data["Price"]>300000).astype(int)

x=data[["Area","Bedrooms","Age"]]
y=data["Expensive"]

X_train,x_test,Y_train,y_test=train_test_split(x,y,test_size=0.1, random_state=42)
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5)
xgb_model.fit(X_train,Y_train)

prediction=xgb_model.predict(x_test)



result=pd.DataFrame(
    {
        "Actual Value:": y_test,
        "Predcited:":prediction
    }
)

print(result)

