# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 09:09:17 2019

Machine Learning -  R-Square with Linear Regression

Evaluation Regression Model Performance
"""

import pandas as pd
import matplotlib.pyplot as plt

# import data
df = pd.read_csv('linear-regression-dataset.csv', sep=';')

# visualize data
plt.scatter(df.deneyim, df.maas)
plt.xlabel('Deneyim')
plt.ylabel('Maas')

#Linear Regression-------
# sklearn library
from sklearn.linear_model import LinearRegression

# linear regression model
linear_reg = LinearRegression()

# convert datasets to arrays (pandas) and reshape for sklearn
x = df.deneyim.values.reshape(-1,1)
y = df.maas.values.reshape(-1,1)

linear_reg.fit(x,y)

# predictions
y_head = linear_reg.predict(x)

plt.plot(x, y_head, color='red')

#%% R-Square
from sklearn.metrics import r2_score

print('r_square score: ', r2_score(y, y_head))