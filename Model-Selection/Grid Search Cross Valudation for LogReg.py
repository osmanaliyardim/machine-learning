# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 11:50:21 2019

@author: Osman Ali Yardım

Machine Learning - Model Selection

Grid Search Cross Valudation for Logistic Regression
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

# %% Grid Search Cross Valudation with KNN
from sklearn.model_selection import GridSearchCV

grid = {'n_neighbors': np.arange(1,50)}
knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn, grid, cv=10) #try all numbers from 1 to 50 to find the best
knn_cv.fit(x, y)

#print K value of KNN algorithm (hyperparameter)
print('---------------')
print('KNN tuned hyperparameter K: ', knn_cv.best_params_) #best value experimentally
print('KNN best accuracy value for tuned value: ', knn_cv.best_score_)

# %% Grid Search Cross Valudation with Logistic Regression
x = x[:100,:]
y = y[:100]

from sklearn.linear_model import LogisticRegression

param_grid = {'C': np.logspace(-3,3,7), 'penalty':['l1','l2']} #if C is greater => overfit, else underfit

logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg, param_grid, cv=10)
logreg_cv.fit(x_train, y_train)

print('---------------')
print('LogReg tuned hyperparameters: ', logreg_cv.best_params_) #best value experimentally
print('LogReg best accuracy value for tuned value: ', logreg_cv.best_score_)

logreg2 = LogisticRegression(C=1, penalty='l1')
logreg2.fit(x_train, y_train)
print('score: ', logreg2.score(x_test, y_test))