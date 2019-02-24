# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 10:33:42 2019

Machine Learning - Decision Tree Classification

Cancerous Cell Prediction / with Kaggle data
"""

#import libraries
import numpy as np
import pandas as pd

#import data
data = pd.read_csv('Decision-Tree-dataset.csv')
print(data.head())

#manipulate data
data.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
print(data.head())

data.diagnosis = [1 if each=='M' else 0 for each in data.diagnosis]

#get features
y = data.diagnosis.values
x_data = data.drop(['diagnosis'], axis=1)

#normalize
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))

#train and test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

#decision tree classification implementation
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)

print('accuracy score of decision tree algorithm: ', dt.score(x_test, y_test))