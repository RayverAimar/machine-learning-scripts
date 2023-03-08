# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 21:23:04 2023

@author: rayver
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Code categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import make_column_transformer
label_encoder_X = LabelEncoder()
X[:,3] = label_encoder_X.fit_transform(X[:,3])
onehotencoder = make_column_transformer((OneHotEncoder(), [3]), remainder = 'passthrough')
X = onehotencoder.fit_transform(X)

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)