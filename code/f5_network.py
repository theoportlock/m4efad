#!/usr/bin/env python
from itertools import permutations
from scipy.stats import spearmanr
from statsmodels.stats.multitest import fdrcorrection
import functions as f
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# load and format data
meta = f.load('meta')
species = f.load('species')
pathways = f.load('pathwaysall')
lipids = f.load('lipids')
extralipids = f.load('extralipids')
lipids = pd.concat([lipids, extralipids], axis=1)
eeg = f.load('eeg')
wolkes = f.load('wolkes')
bayleys = f.load('bayleys')

datasets = {'species':species,
           'pathways':pathways,
           'lipids':lipids,
           'eeg':eeg,
           'wolkes':wolkes,
           'bayleys':bayleys}

# significant changes
shaps = f.load('shaps')
print(shaps.SHAPmean.quantile(0.85))
sigvals = shaps.loc[shaps.SHAPmean.gt(shaps.SHAPmean.quantile(0.85))].index
fshaps = shaps.loc[sigvals]

# Concatenate to full dataframe
#alldata = f.merge(datasets, type='outer')
alldata = f.merge(datasets)

# filter sig
alldf = alldata.droplevel(0, axis=1).loc[:, fshaps.index]

# Calculate correlation
def corr(df):
    combs = list(permutations(df.columns.unique(), 2))
    cors = []
    for comb in combs:
        tdf = pd.concat([df[comb[0]], df[comb[1]]], axis=1).dropna()
        cors.append(spearmanr(tdf[comb[0]], tdf[comb[1]]))
    outdf = pd.DataFrame(cors, index=pd.MultiIndex.from_tuples(combs), columns=['cor','pval'])
    return outdf
outdf = corr(alldf)
outdf = outdf.reset_index().rename(columns={'level_0':'source', 'level_1':'target'}).set_index('source')

# FDR
outdf['qval'] = fdrcorrection(outdf.pval)[1]
foutdf = outdf.loc[outdf.qval.lt(0.05)].sort_values('cor')
f.save(foutdf, 'correlations')

# Connections
salldf = f.stratify(alldf, meta, 'Condition')
f.setupplot(figsize=(1,1))
y='Bacteroides_fragilis'
f.box(salldf, y=y)
f.savefig(y)

y='P108-PWY: pyruvate fermentation to propanoate I'
f.box(salldf, y=y)
f.savefig('pyruvatediffs')

y='Expressive Communication Score'
f.box(salldf, y=y)
f.savefig(y)

x='P108-PWY: pyruvate fermentation to propanoate I'
y='Expressive Communication Score'
sns.regplot(data = salldf, x=x, y=y, scatter_kws={"color": "black", 's':1}, line_kws={"color": "red"})
f.savefig('pyruvatediffs'+y)
