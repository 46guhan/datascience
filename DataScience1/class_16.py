import pymysql as mysql
import pandas as pd
import matplotlib.pyplot as plt
con=mysql.connect(host="localhost",user="root",password="root",database="datascience")
cursor=con.cursor()

cursor.execute("select * from user")
res=cursor.fetchall()


columns=[desc[0] for desc in cursor.description]

df=pd.DataFrame(res,columns=columns)
print(df)

ab=df[df["age"]>30]
be=df[df["age"]<30]

label=["above 30","below 30"]
counts=[len(ab),len(be)]
plt.bar(label,counts,color=["red","blue"])
plt.show()