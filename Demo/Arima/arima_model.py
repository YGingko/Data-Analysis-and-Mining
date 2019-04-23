# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Hai
@time: 2019/4/22 21:45
@desc:
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_model import ARIMA

df = pd.read_excel('discdata.xls')
df = df[df.TARGET_ID == 184]

df['COMPUTER_DISK'] = df.NAME + ':' + df.TARGET_ID.astype(str) + ':' + df.ENTITY + '\\'

df = pd.pivot_table(df, values='VALUE', index='COLLECTTIME', columns='COMPUTER_DISK')
data = df.iloc[:df.shape[0]-5]
xdata = df.iloc[-5:]


###########################################################
# 平稳性检验
###########################################################
diff = 0
adf = ADF(data['CWXT_DB:184:D:\\\\'])
while adf[1] >= 0.05:
    diff = diff + 1
    adf = ADF(df['CWXT_DB:184:D:\\\\'].diff(diff).dropna())

print('The original sequence is smoothed after %s difference, the p value is %s' % (diff, adf[1]))


###########################################################
# 白噪声检验
###########################################################
[[lb], [p]] = acorr_ljungbox(df['CWXT_DB:184:D:\\\\'], lags=1)
if p < 0.05:
    print('The original sequence is a non-white noise sequence, the p value is %s' % p)
else:
    print('The original sequence is a white noise sequence, the p value is %s' % p)

[[lb], [p]] = acorr_ljungbox(df['CWXT_DB:184:D:\\\\'].diff().dropna(), lags=1)
if p < 0.05:
    print('First order difference sequence is a non-white noise sequence, the p value is %s' % p)
else:
    print('First order difference sequence is a white noise sequence, the p value is %s' % p)


###########################################################
# 模型识别
###########################################################
pmax = int(data.shape[0] / 10)
qmax = int(data.shape[0] / 10)
bic_matrix = []
for p in range(pmax + 1):
    tmp = []
    for q in range(qmax + 1):
        try:
            tmp.append(ARIMA(data['CWXT_DB:184:D:\\\\'], (p, 1, q)).fit().bic)
        except:
            tmp.append(None)
    bic_matrix.append(tmp)

bic_matrix = pd.DataFrame(bic_matrix)

p, q = bic_matrix.stack().fillna(np.inf).idxmin()
print('BIC minimum P and q values is %s and %s' % (p, q))


###########################################################
# 模型检验
###########################################################
arima = ARIMA(data['CWXT_DB:184:D:\\\\'], (p, 1, q)).fit()
xdata_pred = arima.predict(typ='levels')
pred_error = (xdata_pred - data['CWXT_DB:184:D:\\\\']).dropna()

lb, p = acorr_ljungbox(pred_error, lags=12)
h = (p < 0.05).sum()
if h > 0:
    print('model ARIMA(1, 1, 1) does not meet white noise test')
else:
    print('model ARIMA(1, 1, 1)  meet white noise test')

arima.forecast()


###########################################################
# 模型评价
###########################################################
pred = arima.forecast(5)[0]
abs_ = (pred - xdata['CWXT_DB:184:D:\\\\']).abs()
mae_ = abs_.mean()
