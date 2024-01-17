#!/usr/bin/env python
import pandas as pd
import seaborn as sns
import numpy as np
import functions as f
import matplotlib.pyplot as plt

meta = f.load('meta')

# bayleys changes
bayleys = f.load('bayleys')
bayleyscondition = f.stratify(bayleys, meta, 'Condition')
chan = f.change(bayleyscondition, analysis=['prevail','diff','mww'])['MalnourishedvsWell-nourished']
f.save(chan, 'bayleyschange')
f.setupplot()
fig, ax= plt.subplots(len(bayleyscondition.columns), 1, figsize=(4,4))
for i, col in enumerate(bayleyscondition.columns):
    f.box(df=bayleyscondition, ax=ax[i], x=bayleyscondition[col], y=bayleyscondition.index)
    ax[i].set_ylabel(col)
    ax[i].set_xlabel(None)
f.savefig('bayleysbox')

# EEG changes
eeg = f.load('eeg')
eegcondition = f.stratify(eeg, meta, 'Condition')
chan = f.change(eegcondition, analysis=['prevail','diff','mww'])['MalnourishedvsWell-nourished']
chan.index = chan.index.str.split(' ', expand=True)
sig = chan['MWW_pval'].unstack().lt(0.05)
fc = chan['meandiff'].unstack()
fc = fc.loc[['Temporal','Frontal','Occipital','Parietal'],['Delta','Theta','Low-Alpha','High-Alpha','Beta','Gamma']]
sig = sig.loc[['Temporal','Frontal','Occipital','Parietal'],['Delta','Theta','Low-Alpha','High-Alpha','Beta','Gamma']]
f.setupplot()
f.heatmap(fc, sig)
f.savefig('eeghm')
chan.index = [' '.join(ind) for ind in chan.index.values]
f.save(chan, 'eegchange')

# wolkes changes
wolkes = f.load('wolkes')
wolkescondition = f.stratify(wolkes, meta, 'Condition')
chan = f.change(wolkescondition, analysis=['prevail', 'diff','mww'])['MalnourishedvsWell-nourished']
f.save(chan, 'wolkeschange')
fig, ax= plt.subplots(len(wolkescondition.columns), 1, figsize=(4,4), sharex=True)
for i, col in enumerate(reversed(wolkescondition.columns.to_list())):
    sns.histplot(data=wolkescondition, ax=ax[i], x=wolkescondition[col], hue=wolkescondition.index, bins=7, element='step', stat='density', common_norm=False, legend=False)
    ax[i].set_ylabel(col)
    ax[i].set_xlabel(None)
f.savefig('wolkesbox')

# correlations
cor, pval = f.corrpair(eeg, pd.concat([wolkes, bayleys], axis=1, join='inner'), min_unique=3)
f.clustermap(cor, pval.lt(0.3))
f.savefig('eegwolkebayleyscorr')
