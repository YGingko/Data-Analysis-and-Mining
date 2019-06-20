# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created by Chocolate on 2019/6/17
"""


import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

data = pd.read_csv('data/train.csv', index_col='id')
x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1],
                                                    test_size=0.3, random_state=0)

rf = RandomForestRegressor()
rf.fit(x_train, y_train)
mean_squared_error(y_test, rf.predict(x_test))

n_estimators = range(10, 80, 1)
params = {'n_estimators': n_estimators}
gsearch = GridSearchCV(estimator=RandomForestRegressor(max_features=None,
                                                       max_depth=8,
                                                       min_samples_split=10,
                                                       min_samples_leaf=4,
                                                       random_state=1),
                       param_grid=params,
                       scoring='neg_mean_squared_error',
                       cv=5,
                       n_jobs=-1)
gsearch.fit(x_train, y_train)
n_estimators = gsearch.best_params_

plt.plot(n_estimators, gsearch.cv_results_['mean_test_score'])
plt.xlabel('n_estimators')
plt.ylabel('nmse')
plt.grid(True)
plt.show()


params = {'max_depth': range(2, 20, 1), 'min_samples_split': range(2, 20, 1)}
gsearch = GridSearchCV(estimator=RandomForestRegressor(n_estimators=64,
                                                       max_features=None,
                                                       min_samples_leaf=1,
                                                       random_state=1),
                       param_grid=params,
                       scoring='neg_mean_squared_error',
                       cv=5,
                       n_jobs=-1)
gsearch.fit(x_train, y_train)
gsearch.best_params_


params = {'min_samples_leaf': range(2, 10, 1), 'min_samples_split': range(2, 20, 1)}
gsearch = GridSearchCV(estimator=RandomForestRegressor(n_estimators=64,
                                                       max_features=None,
                                                       max_depth=14,
                                                       random_state=1),
                       param_grid=params,
                       scoring='neg_mean_squared_error',
                       cv=5,
                       n_jobs=-1)
gsearch.fit(x_train, y_train)
gsearch.best_params_


params = {'max_features': range(1, 8, 1)}
gsearch = GridSearchCV(estimator=RandomForestRegressor(n_estimators=64,
                                                       max_depth=14,
                                                       min_samples_split=12,
                                                       min_samples_leaf=2,
                                                       random_state=1),
                       param_grid=params,
                       scoring='neg_mean_squared_error',
                       cv=5,
                       n_jobs=-1)
gsearch.fit(x_train, y_train)
gsearch.best_params_


gsearch = GridSearchCV(estimator=RandomForestRegressor(n_estimators=64,
                                                       max_features=6,
                                                       max_depth=14,
                                                       min_samples_split=12,
                                                       min_samples_leaf=2,
                                                       random_state=1),
                       param_grid=params,
                       scoring='neg_mean_squared_error',
                       cv=5,
                       n_jobs=-1)
gsearch.fit(x_train, y_train)


rf_best = RandomForestRegressor(n_estimators=64,
                                max_features=6,
                                max_depth=14,
                                min_samples_split=12,
                                min_samples_leaf=2,
                                random_state=1)
rf_best.fit(x_train, y_train)
mean_squared_error(y_test, rf_best.predict(x_test))





