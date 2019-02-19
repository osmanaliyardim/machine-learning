# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 08:15:03 2019

@author: Osman Ali Yardım

Machine Learning - Multiple Linear Regression
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# import data
df = pd.read_csv('multiple-linear-regression-dataset.csv', sep=';')

#Multiple Linear Regression-------
# sklearn library
from sklearn.linear_model import LinearRegression

# convert datasets to arrays (pandas) and reshape for sklearn
x = df.iloc[:,[0,2]].values
y = df.maas.values.reshape(-1,1)

# linear regression model
multiple_linear_reg = LinearRegression()
multiple_linear_reg.fit(x,y)

print("b0: ", multiple_linear_reg.intercept_)
print("b1, b2: ", multiple_linear_reg.coef_)

# predict
print(multiple_linear_reg.predict(np.array([[10,35], [8,35]])))