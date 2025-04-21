""" import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data=pd.read_csv("dataset/employee_unsupervised_dataset.csv")

X=data.drop(["EmployeeID","Department"],axis=1)

scaler=StandardScaler()
Xs=scaler.fit_transform(X)

kmean=KMeans(n_clusters=2,random_state=42)
data["cluster"]=kmean.fit_predict(Xs)
print(data)

plt.scatter(data["Age"],data["Salary"],c=data["cluster"],cmap="Set1",s=200,edgecolors="k")
plt.colorbar(label="Cluster")
plt.show() """

# #Hierarchal clustering =>  unsupervised learning

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram,linkage,fcluster

data=pd.read_csv("dataset/employee_unsupervised_dataset.csv")
X=data.drop(["EmployeeID","Department"],axis=1)

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

linked=linkage(X_scaled,method="ward")

dendrogram(linked,orientation="top",distance_sort="descending")
plt.show()