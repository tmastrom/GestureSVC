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
#print(X.shape)
#print(Y.shape)


# Split the dataset into Testing and Training sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
#print(X_train[1].shape)

# create dataframe from X_train array
dfTrain = pd.DataFrame(X_train)
#print(X_train.shape)

# Perform wavelet transform on training set to get feature vectors
print('Training')
for i,j in dfTrain.iterrows():
    #print(j[0])
    #print(j)
# for testing only do one iteration
    '''if(i >= 0):
        break'''
# initialize arrays for saving sensor values 
    s0 = []
    s1 = []
    s2 = []
    s3 = []
    s4 = []
    s5 = []
    s6 = []
    s7 = []

    count = 0
    for k in j:
        #print(k)
        if( count % 8 == 0 ):
            s0.append(k)
        if( count % 8 == 1 ):
            s1.append(k)
        if( count % 8 == 2 ):
            s2.append(k)
        if( count % 8 == 3 ):
            s3.append(k)
        if( count % 8 == 4 ):
            s4.append(k)
        if( count % 8 == 5 ):
            s5.append(k)
        if( count % 8 == 6 ):
            s6.append(k)
        if( count % 8 == 7 ):
            s7.append(k)

        count += 1

# Perform wavelet transform on the timeseries data for each sensor
    cA, cD = pywt.dwt(s0, 'db1')
    c0 = np.append(cA, cD)
    '''
    print(c0)
    print("cA: {}".format(cA) )
    print("cD: {}".format(cD))
    '''
    cA, cD = pywt.dwt(s1, 'db1')
    c1 = np.append(cA, cD)

    cA, cD = pywt.dwt(s2, 'db1')
    c2 = np.append(cA, cD)

    cA, cD = pywt.dwt(s3, 'db1')
    c3 = np.append(cA, cD)

    cA, cD = pywt.dwt(s4, 'db1')
    c4 = np.append(cA, cD)

    cA, cD = pywt.dwt(s5, 'db1')
    c5 = np.append(cA, cD)

    cA, cD = pywt.dwt(s6, 'db1')
    c6 = np.append(cA, cD)

    cA, cD = pywt.dwt(s7, 'db1')
    c7 = np.append(cA, cD)

    c = np.append(c0, [c1, c2, c3, c4, c5, c6, c7])
    X_train[i] = c
    #print(X_train[i])

# Perform wavelet transform on the test data    
print("Starting testing")
dfTest = pd.DataFrame(X_test)
for i,j in dfTest.iterrows():
    #print(j[0])
    #print(j)

# for testing only do one iteration
    '''if(i >= 0):
        break'''
    s0 = []
    s1 = []
    s2 = []
    s3 = []
    s4 = []
    s5 = []
    s6 = []
    s7 = []

    count = 0
    for k in j:
        #print(k)
        if( count % 8 == 0 ):
            s0.append(k)
        if( count % 8 == 1 ):
            s1.append(k)
        if( count % 8 == 2 ):
            s2.append(k)
        if( count % 8 == 3 ):
            s3.append(k)
        if( count % 8 == 4 ):
            s4.append(k)
        if( count % 8 == 5 ):
            s5.append(k)
        if( count % 8 == 6 ):
            s6.append(k)
        if( count % 8 == 7 ):
            s7.append(k)
        count += 1
    
    cA, cD = pywt.dwt(s0, 'db1')
    c0 = np.append(cA, cD)
    
    print(c0)
    print("cA: {}".format(cA) )
    print("cD: {}".format(cD))

    cA, cD = pywt.dwt(s1, 'db1')
    c1 = np.append(cA, cD)

    cA, cD = pywt.dwt(s2, 'db1')
    c2 = np.append(cA, cD)

    cA, cD = pywt.dwt(s3, 'db1')
    c3 = np.append(cA, cD)

    cA, cD = pywt.dwt(s4, 'db1')
    c4 = np.append(cA, cD)

    cA, cD = pywt.dwt(s5, 'db1')
    c5 = np.append(cA, cD)

    cA, cD = pywt.dwt(s6, 'db1')
    c6 = np.append(cA, cD)

    cA, cD = pywt.dwt(s7, 'db1')
    c7 = np.append(cA, cD)

    c = np.append(c0, [c1, c2, c3, c4, c5, c6, c7])
    X_test[i] = c
    #print(X_train[i])
    
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