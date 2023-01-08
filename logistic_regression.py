# logistics regression works catogorical variable (1 and 0)
# We cant use straight line formaula as the logistic regression limit is 0 to 1, but linear itss infinity
# it is used for classification purposen
# Sigmoid S curve is used and it can convert -infinity to +infinity into 0 to 1 limit

# steps before logistic regression
# 1. Collecting data -- reading the data using pandas
# 2. Analysis  --  mat and seaborn
# 3. Data wrangling -- dummification, missing value tratment, concat, drop
# 4. train and test split  -- sklearn
# 5. classification report  -- score


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Collecting data

input_data=pd.read_csv("titanic_dataset.csv ")
print(input_data.shape)
print(input_data.head())
print("No of passengers travelled",str(len(input_data.index)))

# Analyzing the data

# sns.countplot(x="Survived",data=input_data)
# plt.show()

# sns.countplot(x="Survived",hue="Sex",data=input_data)
# plt.show()

# sns.countplot(x="Survived",hue="Pclass",data=input_data)
# plt.show()

# input_data["Age"].plot.hist()
# plt.show()

# Data Wrangling

print(input_data.isnull())
print(input_data.isnull().sum())

# sns.heatmap(input_data.isnull(),yticklabels=False)
# plt.show()

input_data.drop("Cabin",axis=1,inplace=True)
print(input_data.head())

input_data.dropna(inplace=True)
print(input_data.head())

print(input_data.isnull())

sns.heatmap(input_data.isnull(),yticklabels=False)
plt.show()

print(input_data.isnull().sum())
print(input_data.head())
print(str(len(input_data.index)))

Sex=pd.get_dummies(input_data["Sex"],drop_first=True)
# print(Sex.head())
Pclass=pd.get_dummies(input_data["Pclass"],drop_first=True)
# print(Pclass.head())
Embarked=pd.get_dummies(input_data["Embarked"],drop_first=True)
# print(Embarked.head())

input_data=pd.concat([input_data,Sex,Pclass,Embarked],axis=1)
# print(input_data.head())

input_data.drop(["PassengerId","Pclass","Name","Ticket","Embarked","Sex","Fare","Age"],axis=1,inplace=True)
print(input_data.head())

# Train Test Split

x=input_data.drop("Survived",axis=1).values
y=input_data["Survived"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
prediction=model.predict(x_test)
print(prediction)

# Classification report

from sklearn.metrics import classification_report
a=classification_report(y_test,prediction)
print(a)

from sklearn.metrics import confusion_matrix
b=confusion_matrix(y_test,prediction)
print(b)

from sklearn.metrics import accuracy_score
c=accuracy_score(y_test,prediction)
print(c)