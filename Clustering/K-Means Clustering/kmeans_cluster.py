import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:,[3,4]]

#using elbow method to find optimal number of clusters
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    clusters = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    clusters.fit(X)
    wcss.append(clusters.inertia_)

sns.lineplot(x=range(1,11),y=wcss)
plt.title("The Elbow Method",fontsize=25)
plt.ylabel("WCSS values",fontsize=15)
plt.xlabel("Number of clusters",fontsize=15)
plt.show()

#applying K-means to the dataset
clusters = KMeans(n_clusters=5)
y_kmeans = clusters.fit_predict(X)

#Visualization of the prediction (2-D)
plt.figure(figsize=(13,12))
plt.scatter(x=X.iloc[y_kmeans == 0,0],y=X.iloc[y_kmeans==0,1],s=100,color='red',label='Cluster 1')
plt.scatter(x=X.iloc[y_kmeans == 1,0],y=X.iloc[y_kmeans==1,1],s=100,color='blue',label='Cluster 2')
plt.scatter(x=X.iloc[y_kmeans == 2,0],y=X.iloc[y_kmeans==2,1],s=100,color='green',label='Cluster 3')
plt.scatter(x=X.iloc[y_kmeans == 3,0],y=X.iloc[y_kmeans==3,1],s=100,color='cyan',label='Cluster 4')
plt.scatter(x=X.iloc[y_kmeans == 4,0],y=X.iloc[y_kmeans==4,1],s=100,color='black',label='Cluster 5')
plt.scatter(x=clusters.cluster_centers_[:,0],y=clusters.cluster_centers_[:,1],s=300,color='#a53434',label='Centroids')
plt.title("Clusters of clients",fontsize=35)
plt.xlabel(X.columns[0],fontsize=25)
plt.ylabel(X.columns[1],fontsize=25)
plt.legend()
plt.show()