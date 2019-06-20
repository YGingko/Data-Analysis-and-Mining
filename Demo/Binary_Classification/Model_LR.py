# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created by Chocolate on 2019/6/14
"""

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data_demo.csv')
train, test = train_test_split(df, test_size=0.3, random_state=0)

























