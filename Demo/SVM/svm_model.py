# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Hai
@time: 2019/4/17 16:24
@desc:

"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import svm
from HTools.cm_plot import cm_plot
from sklearn import metrics as met

import pickle

df = pd.read_csv('moment.csv', encoding='gbk')

X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 2:] * 30, df.iloc[:, 0].astype(int),
                                                    test_size=0.3, random_state=0)
model = svm.SVC()
model.fit(X_train, y_train)

pickle.dump(model, open('svm.model', mode='wb'))
# pickle.load(open('svm.model', 'rb'))

cm_train = met.confusion_matrix(y_train, model.predict(X_train))
cm_test = met.confusion_matrix(y_test, model.predict(X_test))

cm_plot(y_train, model.predict(X_train)).show()
cm_plot(y_test, model.predict(X_test)).show()
