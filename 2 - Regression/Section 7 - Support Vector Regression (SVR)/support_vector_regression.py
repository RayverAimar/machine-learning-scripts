# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 18:44:49 2023

@author: rayver
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:, 2:3].values

#y = y.reshape(-1,1)

# Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Fit regression with the dataset
from sklearn.svm import SVR
regression = SVR(kernel="rbf")
regression.fit(X,y)

# Calculate prediction
y_pred = sc_y.inverse_transform(regression.predict(sc_X.transform([[6.5]])).reshape(-1,1))

# Visualization of reesults
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)

plt.scatter(X,y, c="red")
plt.plot(X_grid, regression.predict(X_grid), c="blue")
plt.title("Support Vector Regresion Model")
plt.xlabel("level of Employee")
plt.ylabel("Salary ($)")
plt.show()