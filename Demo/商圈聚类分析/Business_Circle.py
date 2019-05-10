# -*- coding: utf-8 -*-
"""
Created on Thu May  9 11:17:37 2019

@author: Chocolate
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_excel('business_circle.xls')

scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)
df.set_index(u'基站编号', inplace=True)

# 谱系聚类
Z = linkage(df, method='ward', metric='euclidean')
P = dendrogram(Z, 0)
plt.show()

# 层次聚类
k = 3
model = AgglomerativeClustering(n_clusters=k, linkage='ward')
model.fit(df)
r = pd.concat([df, pd.Series(model.labels_, index=df.index)], axis=1)
r.columns = df.columns.tolist() + [u'聚类类别']

style = ['ro-', 'go-', 'bo-']
xlabels = ['工作日上班时间人均停留时间', '凌晨人均停留时间', '周末人均停留时间', '日均人流量']
for i in range(k):
    plt.figure()
    tmp = r[r[u'聚类类别'] == i].iloc[:, :4]
    for j in range(len(tmp)):
        plt.plot(range(1, 5), tmp.iloc[j], style[i])
    plt.xticks(range(1, 5), xlabels, rotation=20)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig('%s%s.png' % ('type_', i))
