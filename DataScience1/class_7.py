#Naive Bayes
"""
Part 1: What is Naive Bayes?
Naive Bayes is a probabilistic machine learning algorithm based on Bayes’ Theorem. It’s called "naive" because it assumes that all features are independent of each other (which is rarely true in reality but often works surprisingly well).


 
P(Y|X) → Posterior: Probability of class Y given features X

P(X|Y) → Likelihood: Probability of features X given class Y

P(Y) → Prior: Probability of class Y

P(X) → Evidence: Probability of features X

📌 Intuition
Imagine you're trying to classify whether an email is spam or not based on keywords. Naive Bayes calculates the probability that the email is spam given the presence (or absence) of words, assuming each word is independent.

🔗 Types of Naive Bayes:
Type	When to Use	Works With
GaussianNB	When features are continuous/numeric	Normal distribution (e.g., age, salary)
MultinomialNB	When features are discrete counts	Word counts in text (e.g., spam detection)
BernoulliNB	When features are binary	Yes/No, 0/1 data (e.g., click/no click)
📊 Part 2: What Kind of Data Can You Use?
Type of Feature	Use with Naive Bayes?	Notes
Categorical (Text)	✅ Yes (after encoding)	Use LabelEncoding or OneHotEncoding
Numerical	✅ Yes	Especially with GaussianNB
Binary (0/1)	✅ Yes	Works great with BernoulliNB
Text Data	✅ Yes	Use MultinomialNB after converting to bag-of-words or TF-IDF
👇 Example Features for Employee Dataset (With GaussianNB):
Feature	Type	Use with NB
Age	Numeric	✅ Yes (GaussianNB)
Department	Categorical	✅ Yes (after encoding)
YearsAtCompany	Numeric	✅ Yes
JobSatisfaction	Ordinal	✅ Yes
Attrition (Label)	Binary	✅ Yes (target)
✅ Summary
Aspect	Naive Bayes
📚 Type	Classifier (Probabilistic)
🧠 Assumption	Features are conditionally independent
💡 Good For	Text classification, simple datasets
📈 Input Type	Numeric, categorical (after encoding), binary
⏱️ Speed	Very fast, low memory
🚫 Not Good For	Highly correlated features, complex relationships

"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


df=pd.read_csv("dataset/employee_supervised_dataset.csv")

df_encode=df.copy()
le=LabelEncoder()
df_encode["Department"]=le.fit_transform(df_encode["Department"])
df_encode["Attrition"]=le.fit_transform(df_encode["Attrition"])


X=df_encode.drop(["EmployeeID","Attrition"],axis=1)
y=df_encode["Attrition"]

# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
# model=GaussianNB()
# model.fit(X_train,y_train)

# y_pred=model.predict(X_test)

# accuracy=accuracy_score(y_test,y_pred)

# print(accuracy)

model=GaussianNB()
model.fit(X,y)

ypred=model.predict(X)
print(ypred)
accuracy=accuracy_score(y,ypred)
print(accuracy)