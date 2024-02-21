# Implementation of Univariate Linear Regression
## AIM:
To implement univariate Linear Regression to fit a straight line using least squares.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Get the independent variable X and dependent variable Y.
2. Calculate the mean of the X -values and the mean of the Y -values.
3. Find the slope m of the line of best fit using the formula. 
<img width="231" alt="image" src="https://user-images.githubusercontent.com/93026020/192078527-b3b5ee3e-992f-46c4-865b-3b7ce4ac54ad.png">
4. Compute the y -intercept of the line by using the formula:
<img width="148" alt="image" src="https://user-images.githubusercontent.com/93026020/192078545-79d70b90-7e9d-4b85-9f8b-9d7548a4c5a4.png">
5. Use the slope m and the y -intercept to form the equation of the line.
6. Obtain the straight line equation Y=mX+b and plot the scatterplot.

## Program:

Program to implement univariate Linear Regression to fit a straight line using least squares.
Developed by: vasanth s
RegisterNumber: 212222110052
### program:
```
import pandas as pd
import numpy as np
dataset=pd.read_csv('/content/student_scores.csv')
print(dataset)

# assigning hours to X & Scores to Y
X=dataset.iloc[:,:1].values
Y=dataset.iloc[:,1].values
print(X)
print(Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)

Y_pred=reg.predict(X_test)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error

plt.scatter(X_train,Y_train,color='green')
plt.plot(X_train,reg.predict(X_train),color='red')
plt.title('Training set (H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show

plt.scatter(X_test,Y_test,color='purple')
plt.plot(X_test,reg.predict(X_test),color='blue')
plt.title('Test set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```
## Output:
![image](https://github.com/vasanth0908/Find-the-best-fit-line-using-Least-Squares-Method/assets/122000018/6ffd55ea-01b4-4f22-9485-c418103405f4)

![image](https://github.com/vasanth0908/Find-the-best-fit-line-using-Least-Squares-Method/assets/122000018/dec43d6e-20b7-48f0-8338-e1101c299e23)

![image](https://github.com/vasanth0908/Find-the-best-fit-line-using-Least-Squares-Method/assets/122000018/2216bc48-cf4e-47e5-92c2-3b3360ca5b3c)

![image](https://github.com/vasanth0908/Find-the-best-fit-line-using-Least-Squares-Method/assets/122000018/372904fb-1c61-4393-a962-0e917b658d4f)

![image](https://github.com/vasanth0908/Find-the-best-fit-line-using-Least-Squares-Method/assets/122000018/1d354d8f-e537-4dd0-9616-92fe78663b8a)

![image](https://github.com/vasanth0908/Find-the-best-fit-line-using-Least-Squares-Method/assets/122000018/d7fd5c06-370e-4339-aa46-ba0a346559a1)

## Result:
Thus the univariate Linear Regression was implemented to fit a straight line using least squares using python programming.
