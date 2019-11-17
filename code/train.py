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

'''for each len 64 vector in the training set
reshape the vector to an 8x8 matrix 
transpose so each row is 8 consecutive readings from a single sensor 
'''
#for i in range(0,X_train.shape[0]):
for i in range(0,1):
    print(X_train[i])

    a = np.reshape(X_train[i], (8,8)).T
    print(a)
    # Perform dwt on each row vector
    for j in range(0,8):
        cA, cD = pywt.dwt(a[j], 'db1')
        a[j] = np.append(cA, cD) # replace the vector with db coefficients

    print("print a",a)
    print("shape of a",a.shape)
    print("shape of X_train" , X_train[i].shape)
    X_train[i] = np.reshape(a, (1,-1))
    print("reshaped training data",X_train[i])

#for i in range(0,X_test.shape[0]):
for i in range(0,1):
    print(X_test[i])
    a = np.reshape(X_test[i], (8,8)).T
    print(a)
    # Perform dwt on each row vector
    for j in range(0,8):
        cA, cD = pywt.dwt(a[j], 'db1')
        a[j] = np.append(cA, cD) # replace the vector with db coefficients

    print("print a",a)
    print("shape of a",a.shape)
    print("shape of X_test" , X_test[i].shape)
    X_test[i] = np.reshape(a, (1,-1))
    print("reshaped training data",X_test[i])

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Test for optimal parameters 
gammas = [0.001, 0.01, 0.1, 1]
degrees = [0, 1, 2, 3, 4, 5, 6]
cs = [0.1, 1, 10, 100, 1000]
for c in cs:
    print('RBF with C {}'.format(c))
    # Create Support Vector Classifier from the Training set
    
    clf = SVC(kernel='rbf',C=c)
    clf.fit(X_train, y_train)

    # Use the classfier to predict Test results
    y_pred = clf.predict(X_test)

    # Create confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print("Test set classification rate: {}".format(np.mean(y_pred == y_test)))
