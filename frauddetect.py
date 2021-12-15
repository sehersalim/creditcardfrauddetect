import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import sklearn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz, export
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#To change the dataset, manually edit the csv file
#Import data
df = pd.read_csv('CreditCard.csv')

#Print total number of missing values, if any
print("Total number of missing values are: ", df.isnull().sum().sum())

#Plot data
class_count = pd.value_counts(df['Class'], sort = True)
class_count.plot(kind = 'bar', rot = 0)
plt.title("Credit Card Transactions")
plt.xticks(range(2), ['Valid', 'Fraud'])
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()

#Check transaction distribution
#Total number of transactions
total = len(df)
print("Total number of credit card transactions: ", total)

#Total number of valid transactions
valid = len(df[df.Class == 0])
print("Total number of valid credit card transactions: ", valid)

#Total number of fraudulent transactions
fraud = len(df[df.Class == 1])
print("Total number of fraudulent credit card transactions: ", fraud)

#Percentage of fraudulent transactions
percent = (fraud*100)/valid
print("Percentage of fraudulent credit card transactions: ", percent)

#Discarding the 'Time' feature
classes = df['Class']
df.drop(['Time', 'Class', 'Amount'], axis = 1, inplace = True)
col = df.columns.difference(['Class'])
minmax = MinMaxScaler()
df = minmax.fit_transform(df)
df = pd.DataFrame(data = df, columns = col)
df = pd.concat([df, classes], axis = 1)

#Model Building

#dependent variable
X = df.drop('Class', axis = 1).values

#independent variable
y = df['Class'].values

#Train and Test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)


#Use Decision Tree
DT = DecisionTreeClassifier(max_depth = 4, criterion = 'entropy')
DT.fit(X_train, y_train)
dt_yhat = DT.predict(X_test)

#Accuracy of DT model
print("Accuracy of the Decision Tree Model is: {}".format(accuracy_score(y_test, dt_yhat)))

#F1-score of DT model
print("F1-score of the Decision Tree Model is: {}".format(f1_score(y_test, dt_yhat)))

#Check confusion matrix
print("Confusion matrix: ")
print(confusion_matrix(y_test, dt_yhat, labels = [0,1]))




