# K - Constant  but always odd
# k nearest neighbour uses pythogorm theorm
# Supervised learning - train and test
# Slow learning algorithm
# Classification
# used in search engine algorithm

# 1. Eucledian distance is used in KNN because it gives the nearest value
# 2. Manhattan distance gives little higher values


# data={"Height":[158,158,158,160,160,163,163,160,163,165,165,165,168,168,168,170,170,170],
#       "Weight":[58,59,63,59,60,60,61,64,64,61,62,65,62,63,66,63,64,68],
#       "Size":["M","M","M","M","M","M","M","L","L","L","L","L","L","L","L","L","L","L"]}
# import pandas as pd
# df=pd.DataFrame(data)
# df.to_csv("knn_data.csv",index=False)

# import pandas as pd
# import numpy as np
#
# data=pd.read_csv("knn_data.csv")
# print(data.head())
#
# k= 3
# x2=161
# y2=61
#
# def dist(ind):
#       x1 = data.iloc[ind]['Height']
#       y1 = data.iloc[ind]['Weight']
#       return ((x2-x1)**2+(y2-y1)**2)**0.5
#
# distance=list(map(dist,data.index))
#
# result=list(zip(data["Size"],distance))
# size=sorted(result,key=lambda x : x[1])[:k]
# knvalues=list(map(lambda x: x[0],size))
# op=max(set(knvalues),key=knvalues.count)
#
# print(f"The shirt size for height {x2}cm and weight {y2}kg is '{op}'")


# import pandas as pd
# import numpy as np
#
# data=pd.read_csv("knn_dataset.csv")
# print(data.head())
#
# k= 3
# x2=161
# y2=61
#
# def dist(ind):
#       x1 = data.iloc[ind]['Height']
#       y1 = data.iloc[ind]['weight']
#       return ((x2-x1)**2+(y2-y1)**2)**0.5
#
# distance=list(map(dist,data.index))
#
# newdf=data.assign(Distance=np.array(distance))
# newdf.sort_values(by="Distance")
# knvalues=newdf["size"][:k].tolist()
# op=max(set(knvalues),key=knvalues.count)
#
# print(f"The shirt size for height {x2}cm and weight {y2}kg is '{op}'")
# ----------------------------------------------------------------------------------------------------------------

# import pandas as pd
# data=pd.read_csv("knn_dataset.csv")
# x_train=data[["Height","weight"]].values
# y_train=data["size"].values
# from sklearn.neighbors import KNeighborsClassifier
# model=KNeighborsClassifier()
# model.fit(x_train,y_train)
# y_test=[[161,66]]
# prediction=model.predict(y_test)
# print(prediction)

# -----------------------------------------------------------------------------------

# import pandas as pd
#
# data = pd.read_csv("knn_dataset.csv")
# x = data[["Height", "weight"]].values
# y = data["size"].values
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
# model = KNeighborsClassifier()
# model.fit(x_train, y_train)
# y_test = [[161, 66]]
# prediction = model.predict(y_test)
# print(prediction)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
data=pd.read_csv('knn_dataset.csv')
a=data.drop('size',axis=1)
b=data['size']
xtrain,xtest,ytrain,ytest=train_test_split(a,b,test_size=0.2,random_state=1)
print(ytest)
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(xtrain,ytrain)
print(xtest)
print(knn.predict(xtest))
