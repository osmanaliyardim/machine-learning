# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 08:29:18 2019

@author: Osman Ali Yardım

Machine Learning - Tree Regression
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# import data
df = pd.read_csv('decision-tree-regression-dataset.csv', sep=';', header=None)

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

# %% decision tree regression
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(x,y)

print(tree_reg.predict(5.5))

x_ = np.arange(min(x), max(x), 0.01).reshape(-1,1) # to cover accurate results (splits)

y_head = tree_reg.predict(x_)

# %% visualize
plt.scatter(x, y, color='red')
plt.plot(x_, y_head, color='green')
plt.xlabel('Tribune Level')
plt.ylabel('Price')
plt.show()