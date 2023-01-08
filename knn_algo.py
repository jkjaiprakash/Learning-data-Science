# import pandas as pd
# import math
# import operator
# data=pd.read_csv("knn_dataset.csv")
# print(data.shape)
# print(data.head())
#
# x=data["Height"]
# y=data["weight"]
# z=data["size"]
#
# n=list(zip(x,y,z))
#
# X,Y=(161,61)
# li=[]
# for x,y,z in n:
#     a=(x-X)**2+(y-Y)**2
#     distance=math.sqrt(a)
#     li.append(([x,y,z],distance))
# ##print(li)
# li.sort(key=operator.itemgetter(1))
# ##print(li)
#
# neighbour=[]
# k=3
# for x in range(k):
#     neighbour.append(li[x][0])
# print(neighbour)
#
# votes={}
# for x in range(len(neighbour)):
#     response=neighbour[x][-1]
#     if response in votes:
#         votes[response]+=1
#     else:
#         votes[response]=1
# sorted_votes=sorted(votes.items(),key=operator.itemgetter(1),reverse=True)
# print(sorted_votes)
# print("the exact match for given measurement is:",sorted_votes[0][0])

###using algorithm

from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split  
import pandas as pd 
# Loading data 
Data = pd.read_csv("knn_dataset.csv")
n=len(Data["Height"] )
# Create feature and target arrays 
a = Data["Height"].values 
b = Data["weight"].values 

X = a.reshape(n,1) 
y = b.reshape(n,1)  
# Split into training and test set 
X_train, X_test, y_train, y_test = train_test_split( 
             X, y, test_size = 0.2, random_state=1) 
print(y_test)
knn = KNeighborsClassifier(n_neighbors=3) 
  
knn.fit(X_train, y_train) 
  
# Predict on dataset which model has not seen before 
print(knn.predict(X_test))










