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
meta = f.load('meta')
species = f.load('taxo')
pathways = f.load('pathways')
bayley = f.load('bayley')
wolkes = f.load('wolkes')
lipids = f.load('lipids')
psd = f.load('psd')

# Filter metadata
fmeta = f.load('metaonehot')
datasets = {'species':species, 'pathways':pathways, "bayley":bayley, 'wolkes': wolkes, 'lipids':lipids, 'psd':psd}
joined = f.merge(datasets)

# filter those categories that are present in at least 20% of total joined datasets
drop=[]
for col in fmeta.columns:
    minlt20 = fmeta.loc[joined.index, col].value_counts().min() < (fmeta.loc[joined.index, col].shape[0] * 0.15)
    less2cat = fmeta.loc[joined.index, col].nunique() < 2
    if minlt20 or less2cat:
        drop.append(col)
meta = fmeta.drop(drop, axis=1)

# Calculate power
targets = meta.columns
target = targets[3]
output = pd.DataFrame(index=datasets.keys())
pval=True
f.setupplot(figsize=(10,120), agg=False)
for dataset in datasets:
    for target in targets:
        tdf = datasets[dataset].join(meta[target].dropna(), how='inner').set_index(target)
        output.loc[dataset,target] = f.PERMANOVA(tdf, pval=pval)
power = -output.apply(np.log)
f.clustermap(power.T, power.T.gt(-np.log(0.1)), square=True)
f.savefig('power')
