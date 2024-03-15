#!/usr/bin/env python
import functions as f
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

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
anthro = f.load('anthro')

# predict all and plot aucrocs
datasets = {'species':species,
           'pathways':pathways,
           'lipids':lipids,
           'eeg':eeg,
           'wolkes':wolkes,
           'bayleys':bayleys,
           'anthro':anthro}

f.setupplot()
fig, ax = plt.subplots(figsize=(3,3))
dataset = list(datasets.keys())[3]
aucrocs = pd.Series(index=datasets.keys())
for dataset in datasets.keys():
    tdf = f.stratify(datasets[dataset], meta, 'Condition')
    tdf = f.upsample(tdf).sort_index(ascending=False)
    print(tdf)
    a,b,c = f.classifier(tdf)
    aucrocs[dataset] = float(b.split('AUCROC=')[1].split('\n')[0])
    f.aucroc(c, ax=ax)
f.savefig('individualmodelsaucroc')
f.setupplot(figsize=(3,3))
aucrocs.sort_values().plot.barh()
f.savefig('aucrocvals')

# Merged model
datasets.pop('anthro')
normdatasets = {key: f.standard(f.filter(dataset, prevail=0.3)) for key, dataset in datasets.items()}
alldata = f.merge(normdatasets).droplevel(0, axis=1)

tdf = f.stratify(alldata, meta, 'Condition')
tdf = f.upsample(tdf).sort_index(ascending=False)
a,b,c = f.classifier(tdf)
shaps = pd.DataFrame(columns=tdf.columns, index=range(100))
roc = []
aucrocs = []
for i in range(100):
    out = f.classifier(tdf, random_state=i)
    aucrocs.append(float(out[1].split('AUCROC=')[1].split('\n')[0]))
    explainer = shap.TreeExplainer(out[0])
    shaps_values = pd.DataFrame(explainer(tdf).values[:,:,0],index=tdf.index, columns=tdf.columns) 
    shaps.loc[i] = shaps_values.abs().mean()
    roc.append(out[2])
shaps = f.norm(shaps)
shaps = shaps.agg(['mean','std']).T
shaps = shaps.set_axis(['SHAPmean','SHAPstd'], axis=1)
meanroc = pd.concat(roc, axis=1).fillna(1).groupby(axis=1, level=0).mean()
stdroc = pd.concat(roc, axis=1).fillna(1).groupby(axis=1, level=0).std()
fig, ax = plt.subplots()
f.aucroc(meanroc, ax=ax)
ax.fill_between(meanroc.fpr, meanroc.tpr-stdroc.tpr, meanroc.tpr+stdroc.tpr, alpha=0.2)
f.savefig('multiomicsmodel')
print(np.array(aucrocs).mean(), np.array(aucrocs).std())

# For comparison with MWW
specieschange = f.load('specieschange')
pathwayschange = f.load('pathwayschange')
lipidclasschange = f.load('lipidclasschange')
extralipidschange = f.load('extralipidschange')
lipidclasschange = pd.concat([lipidclasschange, extralipidschange], axis=0)
eegchange = f.load('eegchange')
wolkeschange = f.load('wolkeschange')
bayleyschange = f.load('bayleyschange')
allchange = {'species':specieschange,
           'pathways':pathwayschange,
           'lipids':lipidclasschange,
           'eeg':eegchange,
           'wolkes':wolkeschange,
           'bayleys':bayleyschange}
mergedchange = pd.concat(allchange)
mergedchange = mergedchange.reset_index().rename(columns={'level_0':'dataset','level_1':'ID'}).set_index('ID')

# Add shaps
mergedchange['-log2(pval)'] = mergedchange.MWW_pval.apply(np.log).mul(-1)
plotdf = mergedchange.join(shaps)

# plot difference
f.setupplot()
fig, ax = plt.subplots()
sns.scatterplot(data = plotdf, x='-log2(pval)', y='SHAPmean', hue='dataset', s=15, ax=ax)
ax.set_ylabel('SHAP_value')
ax.set_xlabel('-log10(p-value)')
print(plotdf.SHAPmean.quantile(0.85))
print('corr =', plotdf.corr()['SHAPmean']['-log2(pval)'])
sigvals = plotdf.loc[plotdf.SHAPmean.gt(plotdf.SHAPmean.quantile(0.85))].index
sig = plotdf.loc[sigvals]
[ax.text(sig.loc[i,'-log2(pval)'], sig.loc[i,'SHAPmean'], s=i) for i in sig.index]
f.savefig('compareshap')
f.save(plotdf, 'shaps')

# infer direction
shaps = f.load('shaps')
shaps['meandiff'] = shaps['Malnourishedmean'] - shaps['Well-nourishedmean']
shaps['percentdiff'] = shaps['Malnourishedmean'].div(shaps['Well-nourishedmean']).mul(100).sub(100)
f.save(shaps, 'shaps')
