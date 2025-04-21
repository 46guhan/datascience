#skewness

#normal distribution [mean==median==mode]
"""
import pandas as pd
from scipy.stats import skew
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv("dataset/supermarket.csv")

numdata=data.select_dtypes(include=["number"])
skewness=numdata.apply(skew)
print(skewness)

for col,skews in skewness.items():
    if skews>0:
      print(f"{col} is a positive skewed")
    elif skews<0:
       print(f"{col} is a negative skewed")
    else:
       print(f"{col} is normally distributted")

sns.barplot(x=skewness.index,y=skewness.values,palette="coolwarm")
plt.show()

""" 

#kurtosis 
"""
import pandas as pd
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv("dataset/supermarket.csv")
numdata=data.select_dtypes(include=["number"])

kurt=numdata.apply(kurtosis)

for col,k in kurt.items():
    if k>3:
        print(col,"this is heavy tails")
    elif k<3:
        print(col,"this is light tails")
    else:
        print(col,"noraml tails") """

#probability theory
#heads or tails
 
import random
import matplotlib.pyplot as plt
def cointoss():
    if(random.randint(0,1)==0):
        return "Heads"
    else:
        return "tails"

def sim():
    n=1000
    result={"Heads":0,"tails":0}
    for _ in range(n):
        result[cointoss()]+=1
    
    print("Head toss : ",result["Heads"])
    print("Tail toss : ",result["tails"])

    head=result["Heads"]/n
    tail=result["tails"]/n

    print("Head prob : ",head)
    print("Tail prob : ",tail)

    labels=result.keys()
    counts=result.values()
    plt.bar(labels,counts,color=["blue","skyblue"])
    plt.show()
sim() 
