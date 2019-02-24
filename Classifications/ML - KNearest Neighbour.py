# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 09:17:58 2019

@author: Osman Ali YARDIM

Machine Learning - Logistic Regression Implementation

Lower Back Pain Symptoms Dataset / with Kaggle data
"""

# initialize libraries
import numpy as np # linear algebra
import matplotlib.pyplot as plt
import pandas as pd # data processing

import seaborn as sns # visualization lib

# read data
data = pd.read_csv('Dataset_spine.csv')

data.head() # show the first 5 elements
# Here, we can see an unwanted column 'Unnamed: 13'

# drop the column which is unwanted
data.drop(['Unnamed: 13'], axis=1, inplace=True)
data.head()

# check types
data.info()
data.describe()
# Here, we can see that we have to change object type to int or category

# To change object type to int or category, we have to change our values (Normal : 0, Abnormal : 1)
# Let's use list comprehension
data.Class_att = [0 if each == 'Normal' else 1 for each in data.Class_att]
data.head()

# Patient / Healthy difference visualization
sns.countplot(x='Class_att', data=data)

# Create features
y = data.Class_att
x_data = data.drop(['Class_att'], axis=1)

# Normalization to prevent missings/denials
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values

# KNN Implementation
# x: features
# y: target variables(normal, abnormal)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
x,y = data.loc[:,data.columns != 'Class_att'], data.loc[:,'Class_att']
knn.fit(x,y)
prediction = knn.predict(x)
print('Prediction: {}'.format(prediction))

# Train and Test Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 1)
knn = KNeighborsClassifier(n_neighbors = 3)
x,y = data.loc[:,data.columns != 'Class_att'], data.loc[:,'Class_att']
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
print('With KNN (K=3) accuracy is: ',knn.score(x_test,y_test)) # accuracy

# Model complexity
neig = np.arange(1, 25)
train_accuracy = []
test_accuracy = []

# Loop over different values of k
for i, k in enumerate(neig):
    # k from 1 to 25(exclude)
    knn = KNeighborsClassifier(n_neighbors=k)
    # Fit with knn
    knn.fit(x_train,y_train)
    #train accuracy
    train_accuracy.append(knn.score(x_train, y_train))
    # test accuracy
    test_accuracy.append(knn.score(x_test, y_test))

# Plot/Visualization
plt.figure(figsize=[13,8])
plt.plot(neig, test_accuracy, label = 'Testing Accuracy')
plt.plot(neig, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.title('Value VS Accuracy')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(neig)
plt.savefig('graph.png')
plt.show()
print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))