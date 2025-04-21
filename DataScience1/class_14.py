with open("data.txt","r") as file:
    text=file.readlines()
    print(text)
import json
import pandas as pd

tab=pd.read_csv("data_tab.txt",sep="\t")
print(tab)


with open("data.json","r") as file:
    data=[json.loads(lines) for lines in file]
    print(data)