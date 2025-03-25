'''What is machine learning? Machine learning is a branch of artificial intelligence (AI) 
and computer science which focuses on the use of data and algorithms to imitate the way that humans learn, 
gradually improving its accuracy.'''

'''Data Science is all about generating insights from raw data. 
This can be achieved by exploring data at a very granular level and understanding the trends. 
Machine learning finds hidden patterns in the data and generates insights that help organizations solve 
the problem.'''




#Naive bayes algorithm

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Load the dataset
df = pd.read_csv("dataset/Tennisdataset1.2.csv")

# Separate features (X) and target variable (y)
X = df.drop(columns=['enjoysport']).values  # Features
y = df['enjoysport'].values  # Target variable

# Encode categorical features to numerical values
label_encoders = [LabelEncoder() for _ in range(X.shape[1])]
for i in range(X.shape[1]):
    X[:, i] = label_encoders[i].fit_transform(X[:, i])

gnb = GaussianNB()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)


""" inval=input()
vowels=('a','e','i','o','u','A','E','I','O','U')
for ch in inval:
    if ch in vowels:
        print(ch) """