# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Hai
@time: 2019/4/16 15:13
@desc:

"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv('air_data.csv', encoding='utf-8')

explore = df.describe(percentiles=[], include='all').T
explore['null'] = df.shape[0] - explore['count']    # Count the number of null value in each feature

df = df[df['SUM_YR_1'].notnull() & df['SUM_YR_2'].notnull()]
index1 = df['SUM_YR_1'] != 0
index2 = df['SUM_YR_2'] != 0
index3 = (df['SEG_KM_SUM'] == 0) & (df['avg_discount'] == 0)
df = df[index1 | index2 | index3]


############################################################################
# 航空公司LRFMC模型
# L: 会员入会时间距观测窗口结束的月数
# R: 客户最近一次乘坐公司飞机距观测窗口结束的月数
# F: 客户在观测窗口内乘坐公司飞机的次数
# M: 客户在观测窗口内累计的飞行里程
# C: 客户在观测窗口内乘坐舱位所对应的折扣系数的平均值
############################################################################

cols = ['LOAD_TIME', 'FFP_DATE', 'LAST_TO_END', 'FLIGHT_COUNT', 'SEG_KM_SUM', 'avg_discount']

data = pd.DataFrame()
data['L'] = np.round((pd.to_datetime(df['LOAD_TIME']) - pd.to_datetime(df['FFP_DATE'])).dt.days / 30, 2)
data['R'] = np.round(df['LAST_TO_END'] / 30, 2)
data['F'] = df['FLIGHT_COUNT']
data['M'] = df['SEG_KM_SUM']
data['C'] = df['avg_discount']

ss = StandardScaler()
zdata = ss.fit_transform(data)

kmodel = KMeans(n_clusters=5, n_jobs=4)
kmodel.fit(zdata)

kmodel.cluster_centers_
kmodel.labels_

from HTools.cluster_plot import cluster_plot
cluster_plot(data, kmodel.cluster_centers_).show()