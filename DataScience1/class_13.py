""" from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("dataset/dim_reduction_dataset.csv")
X=df.drop(columns=["target"])

xs=StandardScaler().fit_transform(X)

pca=PCA(n_components=2)
xp=pca.fit_transform(xs)

plt.scatter(xp[:,0],xp[:,1])
plt.show() """


#Association rule mining

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori,association_rules

df=pd.read_csv("dataset/supermarket_transactions.csv")
transactions=df["Items"].apply(lambda x:x.split(","))

te=TransactionEncoder()
te_data=te.fit_transform(transactions)
df_encoded=pd.DataFrame(te_data,columns=te.columns_)

freq=apriori(df_encoded,min_support=0.05,use_colnames=True)
rule=association_rules(freq,metric="lift",min_threshold=1)

print(rule[["antecedents","consequents","support","confidence","lift"]].sort_values(by="lift",ascending=False).head(10))