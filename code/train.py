import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os
import pywt

# Load the data 
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

# create dataframe from X_train array
dfTrain = pd.DataFrame(X_train)

# for each len 64 vector in the training set
# reshape the vector to an 8x8 matrix 
# transpose so each row is 8 consecutive readings from a single sensor 
for i in range(0,X_train.shape[0]):
    #print(X_train[i])
    a = np.reshape(X_train[i], (8,8)).T
    #print(a)
    # Perform dwt on each row vector
    for j in range(0,8):
        cA, cD = pywt.dwt(a[j], 'db1')
        a[j] = np.append(cA, cD) # replace the vector with db coefficients
    #print("print a",a)
    #print("shape of a",a.shape)
    #print("shape of X_train" , X_train[i].shape)
    X_train[i] = np.reshape(a, (1,-1))
    #print("reshaped training data",X_train[i])

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

# Parameter Tuning
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
'''
gammas = [0.001, 0.01, 0.1, 1]
cs = [0.01, 0.1, 1, 10, 100]
param_grid = {'C': cs, 'gamma' : gammas}

print('Starting Grid Search')
clf = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5,
                       scoring='accuracy')
clf.fit(X_train, y_train)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
'''
# Optimal paramters found using gridsearch
c = 10
g = 0.01

# Create Support Vector Classifier from the Training set
clf = SVC(kernel='rbf',C=c, gamma=g)
clf.fit(X_train, y_train)


### Classification of test set starts here 
import time 

start_time = time.time()
for i in range(0,X_test.shape[0]):
    a = np.reshape(X_test[i], (8,8)).T
    # Perform dwt on each row vector
    for j in range(0,8):
        cA, cD = pywt.dwt(a[j], 'db1')
        a[j] = np.append(cA, cD) # replace the vector with db coefficients
    X_test[i] = np.reshape(a, (1,-1))
X_test = sc.transform(X_test)
# Use the classfier to predict Test results
y_pred = clf.predict(X_test)
print("--- %s seconds ---" % (time.time() - start_time))

# Create confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("Test set classification rate: {}".format(np.mean(y_pred == y_test)))
