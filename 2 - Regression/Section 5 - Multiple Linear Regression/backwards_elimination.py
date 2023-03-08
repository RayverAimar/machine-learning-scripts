# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 09:35:06 2023

@author: rayver
"""

import statsmodels.api as sm

def backwards_elimination(X, y, SL):
    arr = [ i for i in range(len(X[0]))]
    endwhile = False
    while not endwhile:
        X_opt = X[:,arr]
        regression_OLS = sm.OLS(endog=y, exog=X_opt.tolist()).fit()
        values = regression_OLS.pvalues
        max_p_value = max(values)
        if max_p_value > SL:
            arr.pop(values.tolist().index(max_p_value))
        else:
            endwhile = True    
    print(regression_OLS.summary())
    return X_opt