#Logic: Ye ek aisi seedhi line ($y = mx + c$) dhoondta hai jo data points ke darmiyan "Residuals" (error) ko minimize kare.
#Use Case: Price prediction, Salary prediction.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#####################################################
## Read CSV file
data=pd.read_csv("house_data.csv")
print(data.head())
#######################################################
##Make input coloumn
x=data[["Area","Bedrooms","Age"]]
##Make output/target coloumn
y=data["Price"]
########################################################
##Split data 90% Training and 10 Testing
X_train, x_test, Y_train, y_test=train_test_split(x,y,test_size=0.1,random_state=42)  ##random state taky spliting same rhy
#########################################################
##Initialize Model
lr=LinearRegression()
## Train model
lr.fit(X_train,Y_train)
#########################################################
prediction=lr.predict(x_test)
print(prediction)
print(lr.coef_)
print(lr.intercept_)
#########################################################
results = pd.DataFrame({
    "Actual Price": y_test,
    "Predicted Price": prediction
})
print(results)
##R^^2 
Train_Score=lr.score(X_train,Y_train)
print(Train_Score)
Test_Score=lr.score(x_test,y_test)
print(Test_Score)
