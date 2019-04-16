# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 21:41:53 2019

@author: Hai
"""

import pandas as pd

df = pd.read_excel('model.xls')

from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.3, random_state=0)


from keras.models import Sequential # 神经网络初始化函数
from keras.layers.core import Dense, Activation # 神经网络层函数、激活函数

netfile = 'net.model'

net = Sequential()
net.add(Dense(10, input_dim=3))
net.add(Activation('relu'))
net.add(Dense(1, input_dim=10))
net.add(Activation('sigmoid'))
net.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

net.fit(train.iloc[:, :3], train.iloc[:, 3], epochs=1000, batch_size=1)
net.save_weights(netfile)

predict_result = net.predict_classes(train.iloc[:, :3]).reshape(len(train))
net.evaluate(train.iloc[:, :3], train.iloc[:, 3])


from HTools.cm_plot import cm_plot
cm_plot(train.iloc[:, 3], predict_result).show()


from sklearn import metrics
import matplotlib.pyplot as plt

predict_result = net.predict(test.iloc[:, :3]).reshape(len(test))
fpr, tpr, thresholds = metrics.roc_curve(test.iloc[:, 3], predict_result, pos_label=1)
plt.plot(fpr, tpr, linewidth=2, label='ROC of LM')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.ylim(0, 1.05)
plt.xlim(0, 1.05)
plt.legend(loc=4)
plt.show()

print('AUC : ', metrics.auc(fpr, tpr))
print('KS : ', max(tpr - fpr))
