import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os
'''
df0 = pd.read_csv("../input/0.csv", names=['Sensor1','Sensor2',
                                            'Sensor3','Sensor4',
                                            'Sensor5','Sensor6',
                                            'Sensor7','Sensor8', 'Class']) # rock
'''
df0 = pd.read_csv("../input/0.csv", header=None) # rock
df1 = pd.read_csv("../input/1.csv", header=None) # scissors
df2 = pd.read_csv("../input/2.csv", header=None) # paper
df3 = pd.read_csv("../input/3.csv", header=None) # ok
df = pd.concat([df0, df1, df2, df3])

X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

print(X.shape)
print(Y.shape)

'''
df0_x = df0.iloc[:, :-1].values
df0_y = df0.iloc[:, -1].values

#df0 = pd.DataFrame({'x': range(0,8), 'y1': })

print(df0_x.shape)
print(df0_y.shape)

print(df0_x[0])
s1 = []
s2 = []
s3 = []
s4 = []
s5 = []
s6 = []
s7 = []
s8 = []

for i in range(0, df0_x.shape[1], 8):
    s1.append(df0_x[0][i])
    s2.append(df0_x[0][i+1])
    s3.append(df0_x[0][i+2])
    s4.append(df0_x[0][i+3])
    s5.append(df0_x[0][i+4])
    s6.append(df0_x[0][i+5])
    s7.append(df0_x[0][i+6])
    s8.append(df0_x[0][i+7])
    
print(s1)
print(len(s1))
print(s2)

print(s3)
print(s4)
print(s5)
print(s6)
print(s7)
print(s8)


fig = plt.figure()
ax1 = fig.add_subplot(811)
ax2 = fig.add_subplot(812)
ax3 = fig.add_subplot(813)
ax4 = fig.add_subplot(814)
ax5 = fig.add_subplot(815)
ax6 = fig.add_subplot(816)
ax7 = fig.add_subplot(817)
ax8 = fig.add_subplot(818)

xax = list(range(0,8))
print(len(xax))

ax1.plot(xax, s1,  linestyle='-', marker='')
ax1.title.set_text('Sensor1')
ax2.plot(xax, s2,  linestyle='-', marker='')
ax2.title.set_text('Sensor2')
ax3.plot(xax, s3,  linestyle='-', marker='')
ax3.title.set_text('Sensor3')
ax4.plot(xax, s4,  linestyle='-', marker='')
ax4.title.set_text('Sensor4')
ax5.plot(xax, s5,  linestyle='-', marker='')
ax5.title.set_text('Sensor5')
ax6.plot(xax, s6,  linestyle='-', marker='')
ax6.title.set_text('Sensor6')
ax7.plot(xax, s7,  linestyle='-', marker='')
ax7.title.set_text('Sensor7')
ax8.plot(xax, s8,  linestyle='-', marker='')
ax8.title.set_text('Sensor8')
plt.show()

'''

# Split the dataset into Testing and Training sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

print(X_train[1].shape)
'''
s1, s2, s3, s4, s5, s6, s7, s8 = []
for i in range(0, X_train.shape[1], 8):
    s1.append(df0_x[0][i])
    s2.append(df0_x[0][i+1])
    s3.append(df0_x[0][i+2])
    s4.append(df0_x[0][i+3])
    s5.append(df0_x[0][i+4])
    s6.append(df0_x[0][i+5])
    s7.append(df0_x[0][i+6])
    s8.append(df0_x[0][i+7])

for i in range(0, 10):
    print('before')
    x = X_train[i]
    print(x)
    x = x.reshape(8,8)
    print(x)

    X_train[i] = x.reshape(8,8)

    print('after')
    print(X_train[i])

print(X_train.shape)
'''


# create dataframe from X_train array
dfTrain = pd.DataFrame(X_train)
print(dfTrain.shape)


for i,j in dfTrain.iterrows():
    
    #print(j[0])
    #print(j)
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
        print(k)
    
        if( count % 8 == 0 ):
            s0.append(k)

        count += 1

    if(i >= 0):
        print(s0)
        break
'''
    if( i % 8 == 1 ):
        s0.append(j)


    if( i % 8 == 2 ):
        s0.append(j)

    if( i % 8 == 3 ):
        s0.append(j)


    if( i % 8 == 4 ):
        s0.append(j)


    if( i % 8 == 5 ):
        s0.append(j)

    if( i % 8 == 6 ):
        s0.append(j)

    if( i % 8 == 7 ):
        s0.append(j)'''
        
    


'''
X_train= X_train.reshape(-1, 8, 8)

print(X_train.shape)
'''
'''


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
'''