# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 13:06:55 2023

@author: rayver
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# Split dataset into train/test datasets
'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)
'''

# Fit linear regression with dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

# Fit polynomial regression with dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualization of the results of the Linear Model
plt.scatter(X, y, c = "red")
plt.plot(X, lin_reg.predict(X), c = "blue")
plt.title("Linear Regression Model")
plt.xlabel("Position of the employee")
plt.ylabel("Salary ($)")
plt.show()

# Visualization of the results of the Polynomial Model
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1) # Smooth
plt.scatter(X, y, c = "red")
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), c = "blue")
plt.title("Polynomial Regression Model")
plt.xlabel("Position of the employee")
plt.ylabel("Salary ($)")
plt.show()

# Predict of models
lin_reg.predict([[6.5]])
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))