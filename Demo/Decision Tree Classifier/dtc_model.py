# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Hai
@time: 2019/4/16 11:42
@desc:

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics as met
from HTools.cm_plot import cm_plot
from sklearn.externals import joblib

import matplotlib.pyplot as plt


df = pd.read_excel('model.xls')
train, test = train_test_split(df, test_size=0.3, random_state=0)

treefile = 'DecisionTree.pkl'

tree = DecisionTreeClassifier()
tree.fit(train.iloc[:, :3], train.iloc[:, 3])
predict_result = tree.predict(train.iloc[:, :3])
predict_prob = tree.predict_proba(train.iloc[:, :3])[:, 1]

print('Feature Importance: ', tree.feature_importances_)
print('Accuracy: ', met.accuracy_score(train.iloc[:, 3], predict_result))
fpr, tpr, thresholds = met.roc_curve(train.iloc[:, 3], predict_prob)
print('AUC : ', met.auc(fpr, tpr))
print('KS : ', max(tpr - fpr))

cm_plot(train.iloc[:, 3], predict_result).show()


fpr, tpr, thresholds = met.roc_curve(test.iloc[:, 3], tree.predict_proba(test.iloc[:, :3])[:, 1], pos_label=1)
plt.plot(fpr, tpr, linewidth=2, label='ROC of CART', color='green')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.ylim(0, 1.05)
plt.xlim(0, 1.05)
plt.legend(loc=4)
plt.show()