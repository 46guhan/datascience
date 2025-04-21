
#t-distribution

import pandas as pd
import numpy as np
from scipy import stats

data=pd.read_csv("dataset/data.csv")
data=data["Total"]
mean=np.mean(data)
sem=stats.sem(data)
confidence=0.95
n=len(data)

tdis=stats.t.ppf((1+confidence)/2,df=n-1)
marginerror=tdis*sem

l=mean-marginerror
u=mean+marginerror

print(mean,sem,tdis,l,u,sep="\n")



# #data preprocess

# import pandas as pd
# from sklearn.preprocessing import *

# data=pd.read_csv("dataset/supermarket.csv")

# data["Date"]=pd.to_datetime(data["Date"])
# print(data["Date"])

# data["Time"]=pd.to_datetime(data["Time"],format="%H:%M").dt.time
# print(data["Time"])


# #encode categorical columns to numeric

# cat=["Branch","City","Customer type","Gender","Product line","Payment"]
# label_encode=LabelEncoder()
# for  c in cat:
#     data[c]=label_encode.fit_transform(data[c])
#     print(data[c])

# num=["Unit price","Quantity","Tax 5%","Total","cogs","gross income","Rating"]
# scaler=MinMaxScaler()
# data[num]=scaler.fit_transform(data[num])
# print(data[num])