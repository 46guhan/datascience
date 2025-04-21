#normalization - [mean,mode,median all same value and the same line]
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore

data=pd.read_csv("supermarket.csv")
sample=data["Unit price"]
#zscore
data["zscore"]=zscore(sample)
print(data)


#confidence interval

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

data=pd.read_csv("dataset\\supermarket.csv")

data=data["Total"]

mean=np.mean(data)
sem=stats.sem(data) #standard error

confidence=0.95

ci=stats.t.interval(confidence,len(data)-1,loc=mean,scale=sem)

l,u=ci

print("Mean value ",mean)
print("Standard Error ",sem)
print(ci)

plt.axhline(mean,color="green",linestyle="--",label="Mean")
plt.fill_between([0,1],l,u,color="lightblue",alpha=0.5,label="95% confidence interval")
plt.xlim(0,1)
plt.ylim(min(data)-50,max(data)+50)
plt.legend()
plt.show()

"""# plt.axhline(mean,color="green",linestyle="--",label="Mean")
# plt.fill_between([0,1],l,u,color="lightblue",alpha=0.5,label="95% confidence interval")
# plt.xlim(0,1)
# plt.ylim(min(data)-50,max(data)+50)
# plt.legend()
# plt.show() 


#z-distribution

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv("dataset/supermarket.csv")
total=data["Total"]
mean=total.mean()
std=total.std()

data["z-distri"]=(total-mean)/std

print(mean,std,total,data["z-distri"],sep="\n")


plt.subplot(1,2,1)
sns.histplot(total,kde=True,color="skyblue")
plt.subplot(1,2,2)
sns.histplot(data["z-distri"],kde=True,color="red")
plt.show() 


