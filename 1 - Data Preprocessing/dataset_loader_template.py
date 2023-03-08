# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 09:50:04 2023

@author: rayver
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

# N/A data treatment

from sklearn.impute import SimpleImputer as Imputer
imputer = Imputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

# Code categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
label_encoder_X = LabelEncoder()
X[:,0] = label_encoder_X.fit_transform(X[:,0])

# Dummy Encoding is necessary due to France is not bigger than Spain or viceversa

ct = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder="passthrough")
X = ct.fit_transform(X)

label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)

# Split dataset into train/test datasets

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Variable Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
