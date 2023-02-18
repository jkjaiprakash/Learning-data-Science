import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("Customers.csv")
print(data)

x=data.iloc[:,[3,4]].values
print(x)

from sklearn.cluster import KMeans

inertia=[]
centroids=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init="k-means++",max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    inertia.append(kmeans.inertia_)

plt.plot(range(1,11),inertia,'bx-')
plt.title("Elbow method")
plt.xlabel("number of cluster")
plt.ylabel("inertia")
# plt.show()

kmeans=KMeans(n_clusters=5,init="k-means++",max_iter=300,n_init=10,random_state=0)

y_kmeans=kmeans.fit_predict(x)
centroids = kmeans.cluster_centers_
print(centroids)
print(centroids[:,0])
print(centroids[:,1])
print(x[y_kmeans==0])
print(x[y_kmeans==1])
print(x[y_kmeans==2])
print(x[y_kmeans==3])
print(x[y_kmeans==4])

plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=100,c='red',label='cluster1')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=100,c='blue',label='cluster2')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=100,c='green',label='cluster3')
plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],s=100,c='orange',label='cluster4')
plt.scatter(x[y_kmeans==4,0],x[y_kmeans==4,1],s=100,c='cyan',label='cluster5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label="centroids")
plt.title('cluster of clients')
plt.xlabel('Annual income')
plt.ylabel('spending score')
plt.legend()
plt.show()
















