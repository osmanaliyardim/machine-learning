# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 09:33:44 2019

@author: Osman Ali Yardım

Machine Learning - Principle Component Analysis

with Sklearn iris dataset
"""

from sklearn.datasets import load_iris #sklearn dataset
import pandas as pd

iris = load_iris()

data = iris.data
feature_names = iris.feature_names
y = iris.target

df = pd.DataFrame(data, columns=feature_names)
df['clasS'] = y

x = data

# %% PCA implementation
from sklearn.decomposition import PCA
pca = PCA(n_components=2, whiten=True) #whiten->normalize, n_compenents->reducing value
pca.fit(x)

x_pca = pca.transform(x)

print('varience ratio: ', pca.explained_variance_ratio_) #show principle and second component ratio

print('sum: ', sum(pca.explained_variance_ratio_)) #show loss

# %% 2D Visualization
df['p1'] = x_pca[:,0]
df['p2'] = x_pca[:,1]

color = ['red', 'green', 'blue']

import matplotlib.pyplot as plt
for each in range(3):
    plt.scatter(df.p1[df.clasS == each], df.p2[df.clasS == each], color=color[each], label=iris.target_names[each])

plt.legend()
plt.xlabel('p1')
plt.ylabel('p2')
plt.show()