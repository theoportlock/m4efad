#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Theo Portlock
For project setup
'''
import functions as f
import numpy as np
import pandas as pd

df = pd.read_csv("../data/DhakaBangladeshLEAPE-BayleyCompletes_DATA_LABELS_2023-02-23_1725.csv", index_col=0)

df = df.loc[df['Event Name'].str.startswith('1')].iloc[:, 2:]
df = df.loc[:,df.columns.str.contains('Raw')]
df.columns = df.columns.str.replace(' Raw','')
df = df.astype(int)
df = df.dropna()
df.index = df.index.rename('ID')

f.save(df, 'bayley')
