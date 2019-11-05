import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os
from operator import add

df0 = pd.read_csv("../input/0.csv", header=None) # rock
df1 = pd.read_csv("../input/1.csv", header=None) # scissors
df2 = pd.read_csv("../input/2.csv", header=None) # paper
df3 = pd.read_csv("../input/3.csv", header=None) # ok
df = pd.concat([df0, df1, df2, df3])

# do this for each class 
X = df2.iloc[:, :-1].values
Y = df2.iloc[:, -1].values


# Split the dataset into Testing and Training sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

# create dataframe from X_train array
dfTrain = pd.DataFrame(X_train)

# average for visualization
avgs0 = np.zeros(8)
avgs1 = np.zeros(8)
avgs2 = np.zeros(8)
avgs3 = np.zeros(8)
avgs4 = np.zeros(8)
avgs5 = np.zeros(8)
avgs6 = np.zeros(8)
avgs7 = np.zeros(8)
#print(len(avgs0))

# Perform wavelet transform on training set to get feature vectors
print('Training')
for i,j in dfTrain.iterrows():
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

    
    avgs0 = list( map(add, avgs0, s0) )
    avgs1 = list( map(add, avgs1, s1) )
    avgs2 = list( map(add, avgs2, s2) )
    avgs3 = list( map(add, avgs3, s3) )
    avgs4 = list( map(add, avgs4, s4) )
    avgs5 = list( map(add, avgs5, s5) )
    avgs6 = list( map(add, avgs6, s6) )
    avgs7 = list( map(add, avgs7, s7) )
    
    #print(len(s0))
    #print(avgs0)

#need to divide avgs0/length to get mean
avgs0 = np.true_divide(avgs0,X_train.shape[0])
avgs1 = np.true_divide(avgs1,X_train.shape[0])
avgs2 = np.true_divide(avgs2,X_train.shape[0])
avgs3 = np.true_divide(avgs3,X_train.shape[0])
avgs4 = np.true_divide(avgs4,X_train.shape[0])
avgs5 = np.true_divide(avgs5,X_train.shape[0])
avgs6 = np.true_divide(avgs6,X_train.shape[0])
avgs7 = np.true_divide(avgs7,X_train.shape[0])

# display average waveform for each
fig = plt.figure()
ax = fig.add_subplot(111)
ax0 = fig.add_subplot(811)
ax1 = fig.add_subplot(812)
ax2 = fig.add_subplot(813)
ax3 = fig.add_subplot(814)
ax4 = fig.add_subplot(815)
ax5 = fig.add_subplot(816)
ax6 = fig.add_subplot(817)
ax7 = fig.add_subplot(818)

xax = list(range(0,8))
print(len(xax))

ax0.plot(xax, avgs0,  linestyle='-', marker='')

ax0.set_title('s0', loc='right', pad=-15)
ax1.plot(xax, avgs1,  linestyle='-', marker='')
ax1.set_title('s1', loc='right', pad=-15)
ax2.plot(xax, avgs2,  linestyle='-', marker='')
ax2.set_title('s2', loc='right', pad=-15)

ax3.plot(xax, avgs3,  linestyle='-', marker='')
ax3.set_title('s3', loc='right', pad=-15)
ax4.plot(xax, avgs4,  linestyle='-', marker='')
ax4.set_title('s4', loc='right', pad=-15)
ax5.plot(xax, avgs5,  linestyle='-', marker='')
ax5.set_title('s5', loc='right', pad=-15)
ax6.plot(xax, avgs6,  linestyle='-', marker='')
ax6.set_title('s6', loc='right', pad=-15)
ax7.plot(xax, avgs7,  linestyle='-', marker='')
ax7.set_title('s7', loc='right', pad=-15)

ax0.set_ylim([-1.2, 1.2])
ax1.set_ylim([-1.2, 1.2])
ax2.set_ylim([-1.2, 1.2])
ax3.set_ylim([-1.2, 1.2])
ax4.set_ylim([-1.2, 1.2])
ax5.set_ylim([-1.2, 1.2])
ax6.set_ylim([-1.2, 1.2])
ax7.set_ylim([-1.5, 1.2])

# Turn off axis lines and ticks of the big subplot
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
ax.set_title('Gesture "Paper"')
ax.set_ylabel('Normalized sEMG Voltage')
ax.set_xlabel('Time')

plt.show()