#!/usr/bin/env python
import pandas as pd
import seaborn as sns
import numpy as np
import metatoolkit.functions as f
import matplotlib.pyplot as plt

meta = f.load('meta')

# bayleys changes
bayleys = f.load('bayley')
bayleyscondition = f.stratify(bayleys, meta, 'Condition')
chan = f.load('bayleychange')
fchan = f.filter(chan, query='metadata == "Condition.MAM"')
f.setupplot()
fig, ax= plt.subplots(len(bayleyscondition.columns), 1, figsize=(4,4))
for i, col in enumerate(bayleyscondition.columns):
    f.box(df=bayleyscondition, ax=ax[i], x=bayleyscondition[col], y=bayleyscondition.index)
    ax[i].set_ylabel(col)
    ax[i].set_xlabel(None)
f.savefig('bayleysbox')

# PSD changes
psd = f.load('psd')
chan = f.load('psdchange')
fchan = f.filter(chan, query='metadata == "Condition.MAM"')
fchan.index = fchan.index.str.replace('High.','High_').str.replace('Low.','Low_')
fchan.index = fchan.index.str.split('.', expand=True)
sig = fchan['qval'].unstack().lt(0.25)
fc = fchan['coef'].unstack()
fc = fc.loc[['Temporal','Frontal','Occipital','Parietal'],['Delta','Theta','Low_Alpha','High_Alpha','Beta','Gamma']]
sig = sig.loc[['Temporal','Frontal','Occipital','Parietal'],['Delta','Theta','Low_Alpha','High_Alpha','Beta','Gamma']]
f.setupplot()
f.heatmap(fc, sig)
f.savefig('psdhm')

# wolkes changes
wolkes = f.load('wolkes')
wolkescondition = f.stratify(wolkes, meta, 'Condition')
chan = f.change(wolkescondition, analysis=['prevail', 'diff','mww'])['MalnourishedvsWell-nourished']
chan = chan.loc[:, ~chan.columns.str.contains('prev')]
f.save(chan, 'wolkeschange')
fig, ax= plt.subplots(len(wolkescondition.columns), 1, figsize=(4,4), sharex=True)
for i, col in enumerate(reversed(wolkescondition.columns.to_list())):
    sns.histplot(data=wolkescondition, ax=ax[i], x=wolkescondition[col], hue=wolkescondition.index, bins=7, element='step', stat='density', common_norm=False, legend=False)
    ax[i].set_ylabel(col)
    ax[i].set_xlabel(None)
f.savefig('wolkesbox')

# correlations
cor, pval = f.corrpair(psd, pd.concat([wolkes, bayleys], axis=1, join='inner'), min_unique=3)
f.clustermap(cor, pval.lt(0.3))
f.savefig('eegwolkebayleyscorr')
