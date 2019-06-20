# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created by Chocolate on 2019/6/14
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

df = pd.read_csv('data_demo.csv')
train, test = train_test_split(df, test_size=0.3, random_state=0)


x = np.random.uniform(1, 100, 1000)
y = np.log(x) + np.random.normal(0, .3, 1000)

plt.scatter(x, y, s=1, label='log(x) with noise')

plt.plot(np.arange(1, 100), np.log(np.arange(1, 100)), c='b', label='log(x) true function')
plt.xlabel('x')
plt.ylabel('f(x) = log(x)')
plt.legend(loc='best')
plt.title('A Basic Log Function')
plt.show()






































