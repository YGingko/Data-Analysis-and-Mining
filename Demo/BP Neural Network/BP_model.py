# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Hai
@time: 2019/4/17 17:56
@desc:
"""

import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from sklearn import metrics as met

df = pd.read_excel('original_data.xls', sheet_name='属性规约')
df = df[df['水流量'] > 0]

threshold = pd.Timedelta(minutes=4)
df['发生时间'] = pd.to_datetime(df['发生时间'], format='%Y%m%d%H%M%S')
d = df['发生时间'].diff() > threshold
df['事件编号'] = d.cumsum() + 1

df_train = pd.read_excel('train_neural_network_data.xls')
df_test = pd.read_excel('test_neural_network_data.xls')

X_train = df_train.iloc[:, 5:17]
y_train = df_train.iloc[:, 4]
X_test = df_test.iloc[:, 5:17]
y_test = df_test.iloc[:, 4]

model = Sequential()
model.add(Dense(17, input_dim=11))
model.add(Activation('relu'))
model.add(Dense(10, input_dim=17))
model.add(Activation('relu'))
model.add(Dense(1, input_dim=10))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100, batch_size=1)

predict = model.predict_classes(X_train).reshape(len(X_train))
met.accuracy_score(y_train, predict)
model.evaluate(X_train, y_train)


predict = model.predict_classes(X_test).reshape(len(X_test))
met.accuracy_score(y_test, predict)
