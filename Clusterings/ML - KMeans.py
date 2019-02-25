# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 11:34:03 2019

@author: Osman Ali YARDIM

Machine Learning - KMeans Clustering

With Creating Random Dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% creating a random dataset

# class1
x1 = np.random.normal(25,5,1000) # 25->mean, 5->sigma, 1000->quantity
y1 = np.random.normal(25,5,1000)

# class2
x2 = np.random.normal(55,5,1000)
y2 = np.random.normal(60,5,1000)

# class3
x3 = np.random.normal(55,5,1000)
y3 = np.random.normal(15,5,1000)

x = np.concatenate((x1, x2, x3), axis = 0)
y = np.concatenate((y1, y2, y3), axis = 0)

dictionary = {"x":x, "y":y}

data = pd.DataFrame(dictionary)

# visualization of the dataset (supervised vision)
plt.scatter(x1, y1)
plt.scatter(x2, y2)
plt.scatter(x3, y3)
plt.show()

# visualization of the dataset (unsupervised vision)
#plt.scatter(x1, y1, color="black")
#plt.scatter(x2, y2, color="black")
#plt.scatter(x3, y3, color="black")
#plt.show()

# %% K-Means Clustering Algorithm Implementation
from sklearn.cluster import KMeans
wcss = []

for k in range(1,15): #try k values and find ELBOW out
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)

#visualize data to find ELBOW point (3)    
plt.plot(range(1,15), wcss)
plt.xlabel('number of K')
plt.ylabel('WCSS')
plt.show()

# now our k value is 3, let's create our model
kmeans2 = KMeans(n_clusters = 3)
clusters = kmeans2.fit_predict(data)

data['label'] = clusters

#visualize our prediction to see accuracy
plt.scatter(data.x[data.label==0], data.y[data.label==0], color='orange')
plt.scatter(data.x[data.label==1], data.y[data.label==1], color='green')
plt.scatter(data.x[data.label==2], data.y[data.label==2], color='blue')
plt.scatter(kmeans2.cluster_centers_[:,0], kmeans2.cluster_centers_[:,1], color='black') #too see our centroids
plt.show()