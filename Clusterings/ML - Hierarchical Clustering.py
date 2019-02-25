# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 18:15:55 2019

@author: Osman Ali YARDIM

Machine Learning - Hierarchical Clustering / with Dendrogram

With Creating Random Dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% creating a random dataset

# class1
x1 = np.random.normal(25,5,100) # 25->mean, 5->sigma, 1000->quantity
y1 = np.random.normal(25,5,100)

# class2
x2 = np.random.normal(55,5,100)
y2 = np.random.normal(60,5,100)

# class3
x3 = np.random.normal(55,5,100)
y3 = np.random.normal(15,5,100)

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

# %% building dendogram
from scipy.cluster.hierarchy import linkage, dendrogram

merg = linkage(data, method='ward') #decrease variences with WARD
dendrogram(merg, leaf_rotation=90)
plt.xlabel('data points')
plt.ylabel('euclidian distance')
plt.show()

# %% hierarchical clustering implementation
from sklearn.cluster import AgglomerativeClustering

hierarchical_cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
cluster = hierarchical_cluster.fit_predict(data)

data['label'] = cluster

plt.scatter(data.x[data.label==0], data.y[data.label==0], color='red')
plt.scatter(data.x[data.label==1], data.y[data.label==1], color='green')
plt.scatter(data.x[data.label==2], data.y[data.label==2], color='blue')
plt.show()