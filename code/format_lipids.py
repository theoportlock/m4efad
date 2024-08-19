#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Theo Portlock
For lipid data setup
'''
import pandas as pd
import numpy as np
from itertools import permutations
import functions as f

def prevail(df):
    basemean = df.mean().to_frame('basemean')
    means = df.groupby(level=0).mean().T
    means.columns = means.columns + 'mean'
    baseprevail = df.agg(np.count_nonzero, axis=0).div(df.shape[0]).to_frame('baseprev')
    prevail = df.groupby(level=0, axis=0).apply(lambda x: x.agg(np.count_nonzero, axis=0).div(x.shape[0])).T
    prevail.columns = prevail.columns + 'prev'
    basestd = df.std().to_frame('basestd')
    stds = df.groupby(level=0).std().T
    stds.columns = stds.columns + 'std'
    output = pd.concat([basemean,means,baseprevail,prevail,basestd,stds], join='inner', axis=1)
    return output

# Define mapping
LEAP_COLUMN_MAPPING = {r'1001$':'000', r'3301$':'000', r'3302$':'012', r'3303$':'024'}

# Load and clean baseline_dataset (dataset3)
dfp = pd.read_excel("../data/LEAP_01_&_02_combined_Theo_v03.xlsx", sheet_name='POS', index_col=0).iloc[1:,:]
dfn = pd.read_excel("../data/LEAP_01_&_02_combined_Theo_v03.xlsx", sheet_name='NEG', index_col=0).iloc[1:,:]
metabid = pd.read_excel("../data/LEAP_01_&_02_combined_Theo_v03.xlsx", sheet_name='sample IDs', index_col=0)

dfn.index = dfn.index.str.replace(r'.*_S','', regex=True).str.replace(r'.*_Q','Q', regex=True).str.replace(r'_RT_shift','', regex=True)
dfp.index = dfp.index.str.replace(r'.*_S','', regex=True).str.replace(r'.*_Q','Q', regex=True).str.replace(r'_RT_shift','', regex=True)
dfn = dfn.loc[~dfn.index.str.startswith('Q')]
dfp = dfp.loc[~dfp.index.str.startswith('Q')]
dfn.index = dfn.index.astype(int)
dfp.drop('106B', inplace=True)
dfp.index = dfp.index.astype(int)
dfp = dfp.reset_index().rename(columns={'Unnamed: 1':'treatment', 'index':'Liggins sample'}).set_index(['treatment','Liggins sample'])
dfn = dfn.reset_index().rename(columns={'Unnamed: 1':'treatment', 'index':'Liggins sample'}).set_index(['treatment','Liggins sample'])
dfn.columns = dfn.columns.str.replace('\|.*','', regex=True)
dfp.columns = dfp.columns.str.replace('\|.*','', regex=True)
df = pd.concat([dfp,dfn], join='inner', axis=1)
df = df.groupby(level=0, axis=1).max()
metabid['sample'] = metabid['sample'].str[:-4] + '1001'
metabid['Liggins sample'] = metabid['Liggins sample'].str.extract('(\d+)').astype(int)
metabid.treatment = metabid.treatment.str.upper()
metabid.set_index(['treatment','Liggins sample'], inplace=True)
metab = df.join(metabid['sample'], how='inner').set_index('sample')
df = metab.astype(float)

# Filtering ambiguous assignments
df = df.loc[:,df.columns.str.contains('\:')]
df = df.loc[:,~df.columns.str.contains('nsettled')]
df = df.loc[:,~df.columns.str.contains('named')]
df.index = df.index.to_series().replace(LEAP_COLUMN_MAPPING, regex=True)

# Remove 3yr data and lCC7075-7
df = df.loc[df.index.str[3] !='3']
df = df.loc[df.index.str[3] !='7']
df.index = df.index.str[:-3]
df = df.round(6)

# PREVAILANCE
m = df.columns[0]
for m in df.columns:
    zerovals = df[m].value_counts().sort_values().gt(1)
    zerovals = zerovals.loc[zerovals].index.to_list()
    for v in zerovals:
        df.loc[df[m] == v, m] = 0

prev = 0.1 # filter for those that are present in over 10% of samples
meta = f.load('meta')
st = f.stratify(df, meta, 'Condition')
ch = f.prevail(st)
prevs = ch.loc[:, ch.columns.str.contains('prev')]
df = df.drop(prevs.loc[prevs.lt(prev).any(axis=1)].index, axis=1) # lose 100
richness = (~(df == 0)).sum(axis=1).to_frame('lipid_richness')
richness.index = richness.index.rename('ID')
f.save(richness, 'lipid_richness') # Save richness

# Add pseudovalue for 0s
for m in df.columns:
    minval = df.loc[df[m] != 0, m].min()
    df.loc[df[m] == 0, m] = minval/5
df = df.apply(np.log1p) # APPLY LOG - keep above 0
df.index = df.index.rename('ID')
f.save(df, 'lipids')

# for lipid mapping
lipids = pd.read_csv('../data/lipids.tsv', sep='\t', index_col=0, dtype=object)
abbrev = lipids.dropna(subset='ABBREVIATION').set_index('ABBREVIATION')
abbrev = abbrev.groupby(level=0).first()
df = df.T.join(abbrev.dropna(subset='MAIN_CLASS')['MAIN_CLASS'].groupby(level=0).first(), how='inner').set_index('MAIN_CLASS').T
df = df.groupby(level=0, axis=1).sum()
df.index = df.index.rename('ID')
f.save(df, 'lipid_classes')
