# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 08:22:27 2023

@author: rayver
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,2:3].values

# Fit model with dataset
from sklearn.tree import DecisionTreeRegressor
regression = DecisionTreeRegressor(random_state=0)
regression.fit(X,y)

# Prediction of our models
y_pred = regression.predict([[6.5]])

# Visualization of results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X,y, c="red")
plt.plot(X_grid, regression.predict(X_grid), c="blue")
plt.title('Regression model with Decission Tree')
plt.xlabel('Position of employee')
plt.ylabel('Salary ($')
plt.show()