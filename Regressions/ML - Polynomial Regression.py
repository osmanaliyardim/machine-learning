# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 08:25:10 2019

@author: Osman Ali Yardım

Machine Learning - Polynomial Regression
"""

import pandas as pd
import matplotlib.pyplot as plt

# import data
df = pd.read_csv('polynomial-linear-regression.csv', sep=';')

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

# prediction
y_head = lr.predict(x)
print(lr.predict(10000))

# visualize prediction
plt.plot(x, y_head, color='red', label='linear')

# %%
# polynomial regression => y = b0 + b1*x + b2*x^2 + ... + bn*x^n
from sklearn.preprocessing import PolynomialFeatures
polynomial_regression = PolynomialFeatures(degree=4) # give degree greater numbers to get accurate results

x_polynomial = polynomial_regression.fit_transform(x)

linear_regression2 = LinearRegression()
linear_regression2.fit(x_polynomial, y)

y_head2 = linear_regression2.predict(x_polynomial)

plt.plot(x, y_head2, color='black', label='Polynomial')
plt.legend()
plt.show()