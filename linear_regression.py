# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# data=pd.read_csv("efficiency.csv")
# print(data.shape)
# print(data.head())
#
# x=data["speed(km/hr)"].values
# y=data["Distance(km)"].values
#
# mean_x=np.mean(x)
# mean_y=np.mean(y)
# n=len(x)
#
# numerator=0
# denominator=0
#
# for i in range(n):
#     numerator+=(x[i]-mean_x)*(y[i]-mean_y)
#     denominator+=(x[i]-mean_x)**2
# m=numerator/denominator
# print("m:",m)
# c=mean_y-(m*mean_x)
# print("c:",c)
#
# yp=(m*x)+c
# plt.plot(x,yp,color="#723245",label="Regressionline")
# plt.scatter(x,y,color="#107040",label="scatter")
# plt.xlabel("speed")
# plt.ylabel("km")
# plt.show()
#
# a=0
# b=0
#
# for i in range(n):
#     yp=m*x[i]+c
#     a+=(yp-mean_y)**2
#     b+=(y[i]-mean_y)**2
# r_value=(a/b)*100
# print(r_value)

#
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
# x=x.reshape((n,1))
# print(x)
# a=LinearRegression()
# a.fit(x,y)
# plt.scatter(x,y,c="#456755")
# y_prediction=a.predict(x)
# plt.plot(x,y_prediction,c="#657585")
# accuracy=a.score(x,y)
# print(accuracy)
# plt.show()



