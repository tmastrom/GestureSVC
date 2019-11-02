import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# Load data 
loaded = np.load('train.npz')

X_train = loaded['a']
y_train = loaded['b']
X_test = loaded['c']
y_test = loaded['d']
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# visualize data 
'''
colours = ['red','blue','green','black','orange']

for i in range(0, 5):
    colour = colours[i]
    for j in range(0,7):
        plt.plot(X_train[i][j], color=colour)
        
plt.legend()
plt.show()
'''






# PCA
#from sklearn.decomposition import PCA

# instance of the model retaining 95% of the variance
#pca = PCA(.95)

#pca.fit(X_train) # dimensionality is too high

'''

# Wavelet Transform
import pywt

from sklearn.svm import SVC

# Daubechies coefficients
cA, cD = pywt.dwt(X_train[0][0], 'db1')

print(cA) # [ 10.60660172  -2.12132034 -65.05382387   1.41421356]
print(cD) # [  7.77817459   2.12132034 -26.87005769  -2.82842712]

rock = [] 
paper = []
scissors = []
ok = []
# create vectors for each gesture using the Daubechies transform 
'''
