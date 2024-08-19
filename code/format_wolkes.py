#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Theo Portlock
For project setup
'''
import functions as f
import numpy as np
import pandas as pd

df = pd.read_csv('../data/DhakaBangladeshLEAPE-Wolkes_DATA_LABELS_2023-05-10_1935.csv', index_col=0, header=1)
df = df.loc[df['Unnamed: 2'] == '12 months']
df = df.loc[:, df.columns.str.contains('wolke')]
df.columns = df.columns.str.replace('_bangla','')
df.columns = df.columns.str.replace('_bangle','')
df.columns = df.columns.str.replace('wolke_','')
df = df.loc[df.index.str[3] != '3'] # Remove two samples that are 3yr
df = df.astype(int)
df.columns = df.columns.str.title()
df.index = df.index.rename('ID')

f.save(df, 'wolkes')
