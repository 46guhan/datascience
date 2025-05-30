#text mining for data analysis

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data=pd.read_csv("dataset/yelp.csv")

tf=TfidfVectorizer()
x=tf.fit_transform(data["ReviewText"])
print(x)

tf_data=pd.DataFrame(x.toarray(),columns=tf.get_feature_names_out())
print(tf_data)
kmean=KMeans(n_clusters=2,random_state=42)
data["Cluster"]=kmean.fit_predict(x)

cluster=data["Cluster"].value_counts().sort_index()
plt.bar(cluster.index,cluster.values,color="skyblue")
plt.xticks([0,1])
plt.show()