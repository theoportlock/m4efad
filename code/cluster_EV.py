#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Theo Portlock
'''
import metatoolkit.functions as f
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load data  ------------
fmeta = f.load('categories')
species = f.load('species')
pathways = f.load('pathways')
bayley = f.load('bayley')
wolkes = f.load('wolkes')
lipids = f.load('lipids')
psd = f.minmax(f.load('psd'))

# Filter metadata
datasets = {'species':species,
            'pathways':pathways,
            "bayley":bayley,
            'wolkes': wolkes,
            'lipids':lipids,
            'psd':psd}
joined = f.merge(datasets)

# filter those categories that are present in at least 20% of total joined datasets
drop=[]
for col in fmeta.columns:
    minlt20 = fmeta.loc[joined.index, col].value_counts().min() < (fmeta.loc[joined.index, col].shape[0] * 0.15)
    less2cat = fmeta.loc[joined.index, col].nunique() < 2
    if minlt20 or less2cat:
        drop.append(col)
meta = fmeta.drop(drop, axis=1)
 
# filter less relevant ones
meta = meta.loc[:, ~meta.columns.str.startswith('Washing')]
meta = meta.loc[:, ~meta.columns.str.startswith('Household')]
meta = meta.loc[:, ~meta.columns.str.startswith('Family')]
meta = meta.loc[:, ~meta.columns.str.startswith('Frequency')]
meta = meta.loc[:, ~meta.columns.str.startswith('Father')]
meta = meta.loc[:, ~meta.columns.str.startswith('Listens')]
meta = meta.loc[:, ~meta.columns.str.startswith('Uses')]

# remove duplicates
meta = meta.loc[:, ~meta.columns.str.contains('Condition.Well')]
meta = meta.loc[:, ~meta.columns.str.contains('Delivery_Mode.Vag')]
meta = meta.loc[:, ~meta.columns.str.contains('Home')]
meta = meta.loc[:, ~meta.columns.str.contains('Male')]
meta = meta.loc[:, ~meta.columns.str.contains('Type')]
meta = meta.loc[:, ~meta.columns.str.contains('Place_For')]
meta = meta.loc[:, ~meta.columns.str.contains('Principal')]
meta = meta.loc[:, ~meta.columns.str.contains('Mother')]

# Calculate power
output = pd.DataFrame(index=datasets.keys())
targets = meta.columns
pval=False
for dataset in datasets:
    for target in targets:
        tdf = datasets[dataset].join(meta[target].dropna(), how='inner').set_index(target)
        output.loc[dataset,target] = f.PERMANOVA(tdf, pval=pval)
power = output.copy()

f.setupplot()
f.clustermap(power.T, figsize=(3.5,3.1))
f.savefig('power')

