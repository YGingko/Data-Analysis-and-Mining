# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 15:17:11 2019

@author: haitao
"""

# 拉格朗日插值方法
import pandas as pd
from scipy.interpolate import lagrange

def lagrange_column(col, n, k=5):
    '''
    Interpolate value by dataframe column
    
    Parameters:
    -----------
    col : pandas DataFrame column.

    n: int, position where need interpolate a value.
    
    k: int, count.
    -----------
    '''    
    y = col[list(range(n-k, n)) + list(range(n+1, n+1+k))]
    y = y[y.notnull()]
    return lagrange(y.index, list(y))(n)

def lagrange_df(data, k=5):
    '''
    Plot AUC and KS graph.
    
    Parameters:
    -----------
    data : pandas DataFrame with all vars.

    k: int, count.
    -----------
    '''    
    for c in data.columns:
        for r in range(data.shape[0]):
            if (data[c].isnull())[r]:
                data[c][r] = lagrange_column(data[c], r, k)
    return data


