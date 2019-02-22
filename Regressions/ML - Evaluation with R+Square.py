# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 08:55:22 2019

@author: Osman Ali Yardım

Machine Learning -  R-Square with Random Forest

Evaluation Regression Model Performance
"""

import pandas as pd

# import data
df = pd.read_csv('random-forest-regression-dataset.csv', sep=';', header=None)

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

# %% Prediction
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(x,y)

y_head = rf.predict(x)
print(y_head)

# %% R-Square
from sklearn.metrics import r2_score

print('r_score ', r2_score(y, y_head))