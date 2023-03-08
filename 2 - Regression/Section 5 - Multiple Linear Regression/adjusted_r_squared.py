# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 10:08:16 2023

@author: rayver
"""
import numpy as np
import statsmodels.formula.api as sm
def backwardElimination(X, y, SL):    
    numVars = len(X[0])    
    temp = np.zeros((50,6)).astype(int)    
    for i in range(0, numVars):        
        regressor_OLS = sm.OLS(y, X).fit()        
        maxVar = max(regressor_OLS.pvalues).astype(float)        
        adjR_before = regressor_OLS.rsquared_adj.astype(float)        
        if maxVar > SL:            
            for j in range(0, numVars - i):                
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):                    
                    temp[:,j] = X[:, j]                    
                    X = np.delete(X, j, 1)                    
                    tmp_regressor = sm.OLS(y, X).fit()                    
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)                    
                    if (adjR_before >= adjR_after):                        
                        x_rollback = np.hstack((X, temp[:,[0,j]]))                        
                        x_rollback = np.delete(X_rollback, j, 1)     
                        print (regressor_OLS.summary())                        
                        return X_rollback                    
                    else:                        
                        continue    
    regressor_OLS.summary()    
    return X