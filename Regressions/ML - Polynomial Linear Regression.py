# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 08:25:10 2019

@author: Osman Ali Yardım

Machine Learning - Polynomial Linear Regression
"""

import pandas as pd
import matplotlib.pyplot as plt

# import data
df = pd.read_csv('polynomial-regression.csv')

# convert datasets to arrays (pandas) and reshape for sklearn
x = df.araba_max_hiz.values.reshape(-1,1)
y = df.araba_fiyat.values.reshape(-1,1)

# visualize data
plt.scatter(x,y)
plt.xlabel('Araba Max Hiz')
plt.ylabel('Araba Fiyat')

# Polynomial Linear Regression-------
# sklearn library
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x,y)

# predict
y_head = lr.predict(x)

# visualize prediction
plt.plot(x, y_head, color='red')
plt.show()

print(lr.predict(10000))