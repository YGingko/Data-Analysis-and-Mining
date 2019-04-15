# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 21:41:53 2019

@author: haitao
"""

import pandas as pd

df = pd.read_excel('model.xls')

from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.3, random_state=0)


from keras.models import Sequential # 神经网络初始化函数
from keras.layers.core import Dense, Activation # 神经网络层函数、激活函数

netfile = 'net.model'