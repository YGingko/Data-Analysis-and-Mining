# -*- coding: utf-8 -*-
"""
Created on Tue May  7 14:05:06 2019

@author: Chocolate
"""

##############################################################
# 财政收入 = 地方一般预算收入 + 政府性基金收入
# 地方一般预算收入 = 税收收入 + 非税收收入
# 政府性基金收入 = 
##############################################################


import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 10)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LarsCV, LassoCV, Lasso
from GM11 import GM11
from keras.models import Sequential
from keras.layers.core import Dense, Activation

import matplotlib.pyplot as plt

df = pd.read_csv('data1.csv')

# 描述性分析
print(np.round(df.describe().T, 2))

# 相关性分析
print(np.round(df.corrwith(df['y'], method = 'pearson'), 2))

model = LassoCV(alphas=[1, 0.1, 0.01, 0.001, 0.0005])
model.fit(df.iloc[:, :13], df['y'])
model.coef_
model.alpha_

model = Lasso(alpha=0.1)
model.fit(df.iloc[:, :13], df['y'])
model.coef_

pd.Series(model.coef_, index=df.columns[:-1])


df.index = range(1994, 2014)
df.loc[2014] = None
df.loc[2015] = None
l = ['x1', 'x2', 'x3', 'x4', 'x5', 'x7']
for i in l:
    f = GM11(df[i][list(range(1994, 2014))].values)[0]
    df[i][2014] = f(len(df)-1)
    df[i][2015] = f(len(df))
    df[i] = df[i].round(2)


features = ['x1', 'x2', 'x3', 'x4', 'x5', 'x7']
train = df.loc[list(range(1994, 2014)), features + ['y']].copy()

scaler = StandardScaler()
train = scaler.fit_transform(train)
x_train = train[:, :-1]
y_train = train[:, -1]

model = Sequential()
model.add(Dense(12, input_shape = (6, )))
model.add(Activation('relu'))
model.add(Dense(1, input_shape=(12, )))
model.compile(loss = 'mean_squared_error', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 10000, batch_size = 16)
model.save_weights('1-net.model')


x = (df[features] - scaler.mean_[:-1])/scaler.scale_[:-1]
df['y_pred'] = model.predict(x) * scaler.scale_[-1] + scaler.mean_[-1]

df[['y', 'y_pred']].plot(subplots = True, style = ['b-o', 'r-*'])
plt.show()