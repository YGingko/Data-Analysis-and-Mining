# -*- coding: utf-8 -*-
"""
Created on Fri May 10 11:03:14 2019

@author: Chocolate
"""

import numpy as np
import pandas as pd
from gensim import corpora, models


df = pd.read_csv('meidi_jd.txt', sep='&&', encoding='utf-8', header=None)
df.drop_duplicates(inplace=True)


neg_df = pd.read_csv('meidi_jd_neg.txt', encoding='utf-8', header=None)
pos_df = pd.read_csv('meidi_jd_pos.txt', encoding='utf-8', header=None)
stop_df = pd.read_csv('stoplist.txt', encoding='utf-8', header=None, sep='tipdm')

stop = [' ', ''] + list(stop_df[0])

neg_df[1] = neg_df[0].apply(lambda s: s.split(' '))
neg_df[2] = neg_df[1].apply(lambda x: [i for i in x if i not in stop])
pos_df[1] = pos_df[0].apply(lambda s: s.split(' '))
pos_df[2] = pos_df[1].apply(lambda x: [i for i in x if i not in stop])


# 负面主题分析
neg_dict = corpora.Dictionary(neg_df[2])    # 建立词典
neg_corpus = [neg_dict.doc2bow(i) for i in neg_df[2]]   # 建立语料库
neg_lda = models.LdaModel(neg_corpus, num_topics=3, id2word=neg_dict)   # LDA模型训练
for i in range(3):
    neg_lda.print_topic(i)  # 输出每个主题


# 正面主题分析
pos_dict = corpora.Dictionary(pos_df[2])    # 建立词典
pos_corpus = [pos_dict.doc2bow(i) for i in pos_df[2]]   # 建立语料库
pos_lda = models.LdaModel(pos_corpus, num_topics=3, id2word=pos_dict)   # LDA模型训练
for i in range(3):
    pos_lda.print_topic(i)  # 输出每个主题

