#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Theo Portlock
For project setup
'''
import metatoolkit.functions as f
import numpy as np
import pandas as pd

df = pd.read_csv("../data/DhakaBangladeshLEAPE-EEGProcessedVariable_DATA_LABELS_2023-07-10_1606.csv", index_col=0).iloc[:,:-1]
df = df.loc[df['Event Name'].str.startswith('1')].iloc[:, 2:]
df.columns = df.columns.str.replace(' Alpha','-Alpha')
df = df.dropna()
df.columns = df.columns.str.replace(' $','', regex=True)
df.index = df.index.rename('ID')

f.save(df, 'psd')

