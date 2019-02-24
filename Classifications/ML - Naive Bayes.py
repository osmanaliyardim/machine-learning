# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 10:09:32 2019

Machine Learning - Naive Bayes Classification

Cancerous Cell Prediction / with Kaggle data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import data
data = pd.read_csv('Naive-Bayes-dataset.csv')

M = data[data.diagnosis == 'M']
B = data[data.diagnosis == 'B']

# visualization
plt.scatter(M.radius_mean, M.texture_mean, color='red', label='malignant', alpha=0.3)
plt.scatter(B.radius_mean, B.texture_mean, color='green', label='benign', alpha=0.3)
plt.xlabel('radius')
plt.ylabel('texture')
plt.legend()
plt.show()

# data manipulation
data.info()
data.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)
data.diagnosis = [1 if each == 'M' else 0 for each in data.diagnosis]

y = data.diagnosis.values
x_data = data.drop(['diagnosis'], axis=1)

# normalization
x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data)).values

# train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Naive Bayes Implementation
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(x_train, y_train)

print('accuracy of naive bayes algorithm: ', nb.score(x_test, y_test))