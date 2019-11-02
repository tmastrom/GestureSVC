import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))


rock_dataset = pd.read_csv("../input/0.csv", header=None) # class = 0
scissors_dataset = pd.read_csv("../input/1.csv", header=None) # class = 1
paper_dataset = pd.read_csv("../input/2.csv", header=None) # class = 2
ok_dataset = pd.read_csv("../input/3.csv", header=None) # class = 3

frames = [rock_dataset, scissors_dataset, paper_dataset, ok_dataset]
dataset = pd.concat(frames)
# size = (11678, 65)

# scramble up all the rows
dataset_train = dataset.iloc[np.random.permutation(len(dataset))]
dataset_train.reset_index(drop=True)

X_train = []
y_train = []

for i in range(0, dataset_train.shape[0]):
    row = np.array(dataset_train.iloc[i:1+i, 0:64].values)
    X_train.append(np.reshape(row, (64, 1)))
    y_train.append(np.array(dataset_train.iloc[i:1+i, -1:])[0][0])
    
X_train = np.array(X_train)
y_train = np.array(y_train)

print("np.array")
print(X_train.shape)


# Reshape to flatten vector
X_train = X_train.reshape(X_train.shape[0]*X_train.shape[1], 1)

'''
print("reshape")
print(X_train.shape)
'''

X_train = sc.fit_transform(X_train)
'''
print("fit transform")
print(X_train.shape)
'''
# Reshape again after normalization to (-1, 8, 8)
X_train = X_train.reshape((-1, 8, 8))   
'''print("reshape")
print(X_train.shape)

print("ytrain")
print(y_train.shape)
print(y_train)
'''

print("All Data size X and y")
print(X_train.shape)
print(y_train.shape)

# Splitting Train/Test
X_test = X_train[7700:]
y_test = y_train[7700:]
print("Test Data size X and y")
print(X_test.shape)
print(y_test.shape)

X_train = X_train[0:7700]
y_train = y_train[0:7700]
print("Train Data size X and y")
print(X_train.shape)
print(y_train.shape)

print(X_train[0][0]) # [ 13.   2.   0.  -3. -65. -27.  -1.   3.]
print(y_train[0])

np.savez_compressed('train', a=X_train, b=y_train, c=X_test, d=y_test)


