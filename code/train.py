import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os

df0 = pd.read_csv("../input/0.csv", header=None) # rock
df1 = pd.read_csv("../input/1.csv", header=None) # scissors
df2 = pd.read_csv("../input/2.csv", header=None) # paper
df3 = pd.read_csv("../input/3.csv", header=None) # ok
df = pd.concat([df0, df1, df2, df3])

X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

# Split the dataset into Testing and Training sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Create Support Vector Classifier from the Training set
from sklearn.svm import SVC
clf = SVC(gamma='auto')
clf.fit(X_train, y_train)

# Use the classfier to predict Test results
y_pred = clf.predict(X_test)

# Create confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("Test set classification rate: {}".format(np.mean(y_pred == y_test)))

#np.savez_compressed('train', a=X_train, b=y_train, c=X_test, d=y_test)


