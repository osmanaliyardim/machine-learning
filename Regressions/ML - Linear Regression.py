# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 07:53:52 2019

@author: Osman Ali Yardım

Machine Learning - Linear Regression
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# import data
df = pd.read_csv('linear-regression-dataset.csv', sep=';')

# visualize data
plt.scatter(df.deneyim, df.maas)
plt.xlabel('Deneyim')
plt.ylabel('Maas')
plt.show()

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
b0 = linear_reg.predict(0)
print("b0: ", b0)

b0_ = linear_reg.intercept_
print("b0_: ", b0_)

b1 = linear_reg.coef_
print("b1: ", b1) # slope

# maas = 1663 + 1138 * deneyim
maas11 = 1663 + 1138*11
print(maas11)

print(linear_reg.predict(11))

# visualize the fit line
array = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1)

plt.scatter(x,y)
plt.show()

y_head = linear_reg.predict(array) # maas

plt.plot(array, y_head, color='red')