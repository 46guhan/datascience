#Supervised Learning

#Random Forest

"""

What is Random Forest?
Random Forest is an ensemble learning algorithm that builds a forest of decision trees and combines their outputs to improve performance.

How it Works:
Bootstrapping (Bagging): Random subsets of the dataset are created with replacement.

Tree Building: A decision tree is trained on each subset, using a random subset of features.

Voting/Averaging:

For classification: Majority vote across trees.

For regression: Average of predictions from all trees.

🎯 Use Case (Employee Attrition Prediction)
Let’s say you want to predict if an employee will leave. Random Forest would:

Create 100s of trees using different feature/record samples

Each tree gives a Yes/No prediction

The final output is based on majority voting (Yes or No)

✅ Strengths
Feature	Benefit
✅ High accuracy	More stable than a single tree
✅ Handles missing values	And outliers
✅ Feature importance	Easy to find which features matter most
✅ Works with both categorical and numerical data	After encoding
❌ Limitations
Weakness	Description
❌ Slower than decision trees	Especially with large data
❌ Less interpretable	It’s a “black box” compared to a single tree
❌ Overfitting	Can still overfit with too many trees and depth, though less than regular trees
🛠️ When to Use Random Forest?
Employee attrition prediction

Loan approval

Credit scoring

Stock price movement classification

Disease diagnosis

"""

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score


# df=pd.read_csv("dataset/employee_supervised_dataset.csv")

# df_encode=df.copy()
# le=LabelEncoder()
# df_encode["Department"]=le.fit_transform(df_encode["Department"])
# df_encode["Attrition"]=le.fit_transform(df_encode["Attrition"])


# X=df_encode.drop(["EmployeeID","Attrition"],axis=1)
# y=df_encode["Attrition"]

# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# model=RandomForestClassifier(n_estimators=100,random_state=42)
# model.fit(X_train,y_train)

# y_pred=model.predict(X_test)

# accuracy=accuracy_score(y_test,y_pred)

# print(accuracy) 


#linear regression
#only predict a continues data 

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df=pd.read_csv("dataset/employee_supervised_dataset.csv")

X=df[["Age","EducationLevel","YearsAtCompany","JobSatisfaction","WorkLifeBalance"]]
y=df["Salary"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)
print(y_test,y_pred)

print(mean_squared_error(y_test,y_pred))