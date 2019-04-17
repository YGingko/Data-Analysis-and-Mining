# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Hai
@time: 2019/4/17 17:56
@desc:
"""

import pandas as pd

df = pd.read_excel('original_data.xls', sheet_name='属性规约')
df = df[df['水流量'] > 0]

threshold = pd.Timedelta(minutes=4)
df['发生时间'] = pd.to_datetime(df['发生时间'], format='%Y%m%d%H%M%S')
d = df['发生时间'].diff() > threshold
df['事件编号'] = d.cumsum() + 1


