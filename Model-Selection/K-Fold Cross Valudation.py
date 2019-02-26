# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 11:15:02 2019

@author: Osman Ali Yardım

Machine Learning - Model Selection

K-Fold Cross Valudation
"""

from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()

x = iris.data
y = iris.target

#normalization
x = (x - np.min(x)) / (np.max(x) - (np.min(x)))

#train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# KNN model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3) #k => n_neighbors

# %% K-Fold Cross Valudation for K=10
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=knn, X=x_train, y=y_train, cv=10) #cv => k=10
print('average accuracy: ', np.mean(accuracies))
print('distribution of accuracy: ', np.std(accuracies))

# %% Test
knn.fit(x_train, y_train)
print('test accuracy: ', knn.score(x_test, y_test))