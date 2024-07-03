import pandas as pd
import numpy as np

#reading csv file
a=pd.read_csv(r'C:\Users\Venkatesh\Downloads\iris.csv')

#checking the no. of rows and columns of a file
a.shape

#checking the no. of elements
a.size

#displaying 1st 5 rows
a.head()

#displaying columnwise datatypes
a.info()

#describing data 
a.describe()

#target variable and independent variable
x=a.iloc[:,:-1]
y=a.iloc[:,-1]

#coverting dataframes and series to matrices using .values function
x=a.iloc[:,:-1].values
y=a.iloc[:,-1].values

#to split rows as training and testing with desired ratios
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=9)


#performing KNN algorithm
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=1)
model.fit(xtrain,ytrain)


#checking whether the algorithm is working or not
ypred=model.predict(xtest)

#checking the accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,ypred)*100)

#example testcase
print(model.predict([[7.3,4.3,5.5,1.9]]))
