# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn import svm
import itertools




# Reading the .csv file
data = pd.read_csv("creditcard.csv")
df = pd.DataFrame(data)  # Convert data to a Pandas DataFrame

df.describe()  # Descriptive statistics of the features

print(df.columns)

df_fraud = df[df['Class'] == 1]  # Retrieve fraud data
plt.figure(figsize=(15, 10))
plt.scatter(df_fraud['Time'], df_fraud['Amount'])  # Display fraud amounts over time
plt.title('Scatter plot amount fraud')
plt.xlabel('Time')
plt.ylabel('Amount')
plt.xlim([0, 175000])
plt.ylim([0, 2500])
plt.show()

nb_big_fraud = df_fraud[df_fraud['Amount'] > 1000].shape[0]  # Number of frauds with amount greater than 1000
print('There are only ' + str(nb_big_fraud) + ' frauds where the amount was bigger than 1000 out of ' + str(
    df_fraud.shape[0]) + ' frauds')

number_fraud = len(data[data.Class == 1])
number_no_fraud = len(data[data.Class == 0])
print(
    'There are only ' + str(number_fraud) + ' frauds in the original dataset, even though there are ' + str(
        number_no_fraud) + ' non-frauds in the dataset.')

print("The accuracy of the classifier would be: " + str((284315 - 492) / 284315) +
      " which is the number of correct classifications over the total number of tuples")

df_corr = df.corr()  # Calculate the correlation coefficients between features using the Pearson method

plt.figure(figsize=(15, 10))
plt.imshow(df_corr, cmap="YlGnBu")  # Display the correlation heatmap
plt.colorbar()
plt.xticks(range(len(df_corr.columns)), df_corr.columns, rotation=90)
plt.yticks(range(len(df_corr.columns)), df_corr.columns)
plt.title('Heatmap correlation')
plt.show()

rank = df_corr['Class']  # Retrieve the correlation coefficients between features and the target class
df_rank = pd.DataFrame(rank)
df_rank = np.abs(df_rank).sort_values(by='Class', ascending=False)  # Rank the absolute values of the coefficients in descending order
df_rank.dropna(inplace=True)  # Remove missing data

# Split the data into train and test datasets

# Build the train dataset
df_train_all = df[0:150000]  # Split the original dataset
df_train_1 = df_train_all[df_train_all['Class'] == 1]  # Separate frauds and non-frauds
df_train_0 = df_train_all[df_train_all['Class'] == 0]
print('In this dataset, there are ' + str(len(df_train_1)) + " frauds, so we need to take a similar number of non-frauds")

df_sample = df_train_0.sample(300)
df_train = pd.concat([df_train_1, df_sample])  # Combine frauds with non-frauds
  # Combine frauds with non-frauds
df_train = df_train.sample(frac=1)  # Shuffle the dataset

X_train = df_train.drop(['Time', 'Class'], axis=1)  # Drop the Time and Class columns
y_train = df_train['Class']  # Create the label
X_train = np.asarray(X_train)
y_train = np.asarray(y_train)

# Use all the test dataset to evaluate the model
df_test_all = df[150000:]

X_test_all = df_test_all.drop(['Time', 'Class'], axis=1)
y_test_all = df_test_all['Class']
X_test_all = np.asarray(X_test_all)
y_test_all = np.asarray(y_test_all)

X_train_rank = df_train[df_rank.index[1:11]]  # Select the top ten ranked features
X_train_rank = np.asarray(X_train_rank)

X_test_all_rank = df_test_all[df_rank.index[1:11]]
X_test_all_rank = np.asarray(X_test_all_rank)
y_test_all = np.asarray(y_test_all)

class_names = np.array(['0', '1'])  # Binary labels: Class = 1 (fraud) and Class = 0 (non-fraud)

# Function to plot the confusion matrix
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


classifier = svm.SVC(kernel='linear')  # Use SVM classifier with linear kernel

classifier.fit(X_train, y_train)  # Train the model with the balanced train data

prediction_SVM_all = classifier.predict(X_test_all)  # Predict using the test data

cm = confusion_matrix(y_test_all, prediction_SVM_all)
plot_confusion_matrix(cm, class_names)

print('Our criterion gives a result of ' +
      str(((cm[0][0] + cm[1][1]) / (sum(cm[0]) + sum(cm[1])) + 4 * cm[1][1] / (cm[1][0] + cm[1][1])) / 5))

print('We have detected ' + str(cm[1][1]) + ' frauds out of ' + str(cm[1][1] + cm[1][0]) + ' total frauds.')
print('\nThe probability to detect a fraud is ' + str(cm[1][1] / (cm[1][1] + cm[1][0])))
print("The accuracy is: " + str((cm[0][0] + cm[1][1]) / (sum(cm[0]) + sum(cm[1]))))
