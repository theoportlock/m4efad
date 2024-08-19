#!/usr/bin/env python
import matplotlib.pyplot as plt
import metatoolkit.functions as f
import numpy as np
import pandas as pd
import seaborn as sns
import shap

# load and format data
shaps = f.load('alldatafilterCondition.MAMmeanabsshapsmean')
shaps = shaps.set_axis(['SHAPmean'], axis=1)

# Shaps histogram
f.setupplot()
ax = shaps['SHAPmean'].plot.hist(bins=100)
f.savefig('shaps_hist')

# Rename to R format
shaps.index = shaps.index.str.replace(r'[\[\]\;\/\|\(\)\:\-\ ]','.', regex=True)

# For comparison with maaslin2
taxochange = f.load('taxochangefilter')
pathwayschange = f.load('pathwayschangefilter')
melonchange = f.load('melonchangefilter')
lipidclasseschange = f.load('lipid_classeschangefilter')
psdchange = f.load('psdchangefilter')
wolkeschange = f.load('wolkeschangefilter')
bayleyschange = f.load('bayleychangefilter')
allchange = {'taxo':taxochange,
           'pathways':pathwayschange,
           'lipids':lipidclasseschange,
           'melon':melonchange,
           'psd':psdchange,
           'wolkes':wolkeschange,
           'bayleys':bayleyschange}
mergedchange = pd.concat(allchange)
mergedchange = mergedchange.reset_index().rename(columns={'level_0':'dataset'}).set_index('feature')

# Add shaps
mamchange = mergedchange.copy()
plotdf = mamchange.join(shaps, how='inner')
#plotdf['-log10(qval)'] = plotdf.qval.apply(np.log10).mul(-1)
plotdf['-log10(qval)'] = plotdf.pval.apply(np.log10).mul(-1)

# plot difference
f.setupplot(figsize=(7,7))
ax = sns.scatterplot(data = plotdf.dropna(), x='-log10(qval)', y='SHAPmean', hue='dataset', s=12)
ax.set_ylabel('SHAP_value')
ax.set_xlabel('-log10(qval)')
print(plotdf['SHAPmean'].quantile(0.95))
print('corr = ', plotdf[['SHAPmean','-log10(qval)']].corr().iloc[0,1])
sigvals = plotdf.loc[plotdf['SHAPmean'].gt(plotdf['SHAPmean'].quantile(0.95))].index
sig = plotdf.loc[sigvals]
[ax.text(sig.loc[i,'-log10(qval)'], sig.loc[i,'SHAPmean'], s=i) for i in sig.index]
f.savefig('compareshap')
f.save(plotdf, 'shaps')
