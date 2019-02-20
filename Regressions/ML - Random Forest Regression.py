# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 09:05:26 2019

@author: Osman Ali Yardım

Machine Learning - Random Forest Regression
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# import data
df = pd.read_csv('random-forest-regression-dataset.csv', sep=';', header=None)

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

# %% prediction and visualization
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(x,y)

print(rf.predict(7.5))

x_ = np.arange(min(x), max(x), 0.01).reshape(-1,1)
y_head = rf.predict(x_)

plt.scatter(x, y, color='red')
plt.plot(x_, y_head, color='green')
plt.xlabel('Tribune Level')
plt.ylabel('Price')
plt.show()