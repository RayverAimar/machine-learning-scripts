# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 21:23:04 2023

@author: rayver
"""

# Multiple Linear Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Code categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import make_column_transformer
label_encoder_X = LabelEncoder()
X[:,3] = label_encoder_X.fit_transform(X[:,3])
onehotencoder = make_column_transformer((OneHotEncoder(), [3]), remainder = 'passthrough')
X = onehotencoder.fit_transform(X)

# Avoid the Dummy Variables Trap
X = X[:,1:]

# Split train and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)

# Adjust the Mulple Linear Regression model with the training dataset
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

# Predict results on testing dataset
y_pred = regression.predict(X_test)

# Calculate mean error
errors = [abs(y_pred[i] - y_test[i])/y_test[i] for i in range(len(y_test))]
mean_error = sum(errors)/len(errors)
print(f'Mean error: {mean_error * 100:.2f}%')

# Calculate optimal MLR model using Backwards Elimintation
import statsmodels.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values=X, axis=1)
SL = 0.05

X_opt = X[:,[0,1,2,3,4,5]]
regression_OLS = sm.OLS(endog=y, exog=X_opt.tolist()).fit()
regression_OLS.summary()

X_opt = X[:,[0,1,3,4,5]]
regression_OLS = sm.OLS(endog=y, exog=X_opt.tolist()).fit()
regression_OLS.summary()

X_opt = X[:,[0,3,4,5]]
regression_OLS = sm.OLS(endog=y, exog=X_opt.tolist()).fit()
regression_OLS.summary()

X_opt = X[:,[0,3,5]]
regression_OLS = sm.OLS(endog=y, exog=X_opt.tolist()).fit()
regression_OLS.summary()

X_opt = X[:,[0,3]]
regression_OLS = sm.OLS(endog=y, exog=X_opt.tolist()).fit()
regression_OLS.summary()

# Own backwards elimination function
from backwards_elimination import backwards_elimination
X_opt_2 = backwards_elimination(X, y, SL)


