# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 11:21:33 2019

@author: haitao
"""

import pandas as pd
df = pd.read_excel('Test Data/missing_data.xls', header=None)

import Interpolate as inp
df = inp.lagrange_df(df)
