# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Hai
@time: 2019/4/17 10:55
@desc:

"""

import pandas as pd
from sklearn.cluster import KMeans
from HTools.apriori import *
import time

df = pd.read_excel('data.xls')

k = 4
dis_cols = [x for x in df.columns if x.endswith('系数')]
dis_labels = []
for col in dis_cols:
    print(u'正在进行“%s”的聚类...' % col)
    kmodel = KMeans(n_clusters=k)
    kmodel.fit(df[[col]])
    dis_label = pd.DataFrame(kmodel.cluster_centers_).sort_values(by=0).rolling(2).mean().fillna(0)[0].tolist()
    dis_label = dis_label + [1]
    dis_labels.append(dis_label)

    df[col] = pd.cut(df[col], bins=dis_label, labels=[col[:2] + '_' + str(x) for x in range(k)])


data = df[dis_cols + ['TNM分期']]

start = time.clock()
data = pd.DataFrame(list(map(lambda x: pd.Series(1, index=x[pd.notnull(x)]), data.values))).fillna(0)
end = time.clock()
print(u'\n转换完毕，用时：%0.2f秒' % (end-start))

support = 0.06
confidence = 0.75

start = time.clock()
print(u'\n开始搜索关联规则...')
find_rule(data, support, confidence)
end = time.clock()  # 计时结束
print(u'\n搜索完成，用时：%0.2f秒' % (end-start))

