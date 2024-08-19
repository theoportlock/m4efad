#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Theo Portlock
For project setup
'''
import functions as f
import numpy as np
import pandas as pd

df1 = pd.read_excel('../data/06._LEAP_Child_Anthropometry_for_1_Yrs_MAM (Only Baseline)_06-Mar-2023.xlsx', index_col=0)
df2 = pd.read_excel('../data/07._LEAP-Child_Anthropometry_for_1_Yrs_Normal_16-Feb-2023.xlsx', index_col=0)

df2 = df2.loc[df2.An_Time == 1]
df = pd.concat([df1, df2])
df.DOM = pd.to_datetime(df.DOM)
df.DOB = pd.to_datetime(df.DOB)
df['DaysAfterBirth'] = df.DOM - df.DOB
df.DaysAfterBirth = df.DaysAfterBirth.dt.days
df.index = 'LCC' + df.index.astype(str)
df['SEX'] = df['SEX'].astype(str).map({'1':'Male','2':'Female'})
df.drop(['SEX', 'DaysAfterBirth', 'DOV', 'DOB', 'DOM', 'Missed', 'Group','Length', 'An_Time'], axis=1, inplace=True)
df.index = df.index.rename('ID')

f.save(df, 'anthro')
