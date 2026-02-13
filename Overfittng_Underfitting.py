#1. Overfitting vs Underfitting
#Underfitting (Model is too simple): Model data ke patterns ko samajh hi nahi pa raha. Na training mein acha perform karta hai na testing mein.
#Misal: Aapne 10th grade ka math seekhna hai par aap sirf 2+2 seekh kar ruk gaye.
#Overfitting (Model is too complex): Model ne training data ko "ratta" maar liya hai. Training mein 100% result deta hai par naye data (Test) par fail ho jata hai.
#Misal: Aapne math ke purane papers ke answers yaad kar liye, lekin exam mein jab sawal ki values badli gayin toh aap fail ho gaye.

import pandas as pd
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.model_selection import train_test_split

data=pd.read_csv("house_data.csv")
print(data.head())

x=data[["Area","Bedrooms","Age"]]
y=data["Price"]

X_train, x_test,Y_train,y_test=train_test_split(x,y,test_size=0.1, random_state=42)

lr=LinearRegression()
lr.fit(X_train,Y_train)
predcition=lr.predict(x_test)

result=pd.DataFrame(
    {
        "Actual Value:": y_test,
        "Predcited:":predcition
    }
)

print(result)

#A. L1 Regularization (Lasso)
#Logic: Ye weights ki absolute value ko penalty mein dalta hai. Iski khasiyat ye hai ke ye bekar features ke weights ko bilkul zero kar deta hai.
#Use Case: Jab aapke paas 100 features hon aur aap chahte hon ke sirf kaam ke features bachein (Feature Selection).

lasso=Lasso(alpha=0.1)
lasso.fit(X_train,Y_train)

prediction_lasso=lasso.predict(x_test)

result=pd.DataFrame(
    {
        "Actual Value:": y_test,
        "Predcited:":prediction_lasso
    }
)

print(result)

#B. L2 Regularization (Ridge)
#Logic: Ye weights ke "square" ko penalty mein dalta hai. Ye weights ko zero nahi karta lekin unhe bohot chota (close to zero) kar deta hai.
#Use Case: Jab aap chahte hon ke saare features thora thora contribute karein lekin koi bhi haavi na ho


ridge1=Ridge(alpha=0.5)
ridge1.fit(X_train,Y_train)

prediction_ridge=ridge1.predict(x_test)

result=pd.DataFrame(
    {
        "Actual Value:": y_test,
        "Predcited:":prediction_ridge
    }
)

print(result)