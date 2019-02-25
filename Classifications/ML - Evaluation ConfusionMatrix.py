# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 09:17:25 2019

@author: Osman Ali YARDIM

Machine Learning - Classification Evaluation: Confusion Matrix

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

#random forest classification implementation
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=1)
rf.fit(x_train, y_train)
print('accuracy score of random forest algorithm ', rf.score(x_test, y_test))

#evaluation: confusion matrix
y_pred = rf.predict(x_test)
y_true = y_test

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)

#visualization of confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm, annot=True, linewidth=0.5, linecolor='red', fmt='.0f', ax=ax)
plt.xlabel('y_pred')
plt.ylabel('y_true')
plt.show()