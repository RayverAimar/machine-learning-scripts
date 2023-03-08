# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 17:43:49 2023

@author: rayver
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

# Split dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=1/3, random_state=0)

# No need to scale variables for Simple Linear Regression (specifically salary case)

# Create Simple Linear Regression model with the training dataset
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)


# Predict test dataset 
y_pred = regression.predict(X_test)

# Visualize training results
plt.scatter(X_train, y_train, c="red")
plt.plot(X_train, regression.predict(X_train), c="blue")
plt.title("Salary vs Years of Experience (Training Dataset)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary ($)")
plt.show()

# Visualize testing results
plt.scatter(X_test, y_test, c="red")
plt.plot(X_train, regression.predict(X_train), c="blue")
plt.title("Salary vs Years of Experience (Testing Dataset)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary ($)")
plt.show()