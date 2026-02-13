#SVM (Support Vector Machine)
#Logic: Ye do classes ke beech ek aisi "Deewar" (Hyperplane) banata hai jiska dono taraf se fasla (Margin) sabse zyada ho. Agar data linear na ho, toh ye "Kernel Trick" use karke data ko higher dimension mein le jata hai.
#Use Case: Image recognition, Face detection.
#Coding Methods:
#kernel: 'linear', 'poly', 'rbf' (Non-linear ke liye).
#C: Regularization parameter (Ghaltiyon par kitni saza deni hai).

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

data=pd.read_csv("house_data.csv")
print(data.head())

x=data[["Area","Bedrooms","Age"]]
y=data["Price"]

X_train,x_test,Y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=42)

svm1=SVC(C=1.0,kernel="linear",probability=True)

svm1.fit(X_train,Y_train)

print(svm1.predict(x_test))

result=pd.DataFrame({
    "Actual Value:":y_test,
    "Predicted:":svm1.predict(x_test)
})
print(result)

print(svm1.score(X_train,Y_train))
