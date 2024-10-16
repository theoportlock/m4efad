#!/usr/bin/env python
import matplotlib.pyplot as plt
import metatoolkit.functions as f
import numpy as np
import pandas as pd
import seaborn as sns

def prev(tdf):
    basemean = tdf.mean().to_frame('basemean')
    means = tdf.groupby(level=0).mean().T
    means.columns = means.columns.astype(str) + 'mean'
    baseprevail = tdf.agg(np.count_nonzero, axis=0).div(tdf.shape[0]).to_frame('baseprev')
    prevail = tdf.groupby(level=0, axis=0).apply(lambda x: x.agg(np.count_nonzero, axis=0).div(x.shape[0])).T
    prevail.columns = prevail.columns.astype(str) + 'prev'
    basestd = tdf.std().to_frame('basestd')
    stds = tdf.groupby(level=0).std().T
    stds.columns = stds.columns.astype(str) + 'std'
    output = pd.concat([basemean,means,baseprevail,prevail,basestd,stds], join='inner', axis=1)
    return output

def fc(df, comb=None):
    print(comb)
    outdf = pd.DataFrame(
        df.loc[comb[0]].mean().div(df.loc[comb[1]].mean()),
        columns = ['FC'],
        index = df.columns)
    outdf['Log2FC'] = outdf.FC.apply(np.log2)
    outdf['Log10FC'] = outdf.FC.apply(np.log10)
    return outdf

# Load species
meta = f.load('metaonehot')
df =  f.load('taxo')
df = df.loc[:, df.columns.str.startswith('s')]

## Filter top 50
filttop50 = df.mean().sort_values().tail(50)
df = df.loc[:, filttop50.index]

# taxa changes
df = f.stratify(df, meta, 'Condition.MAM')
df.columns = df.columns.str.replace('s__','').str.replace('_',' ')
prevs = prev(df)
prevdf = prevs[['Falseprev','Trueprev']].stack().reset_index()

# supp figure for species
plotdf = df.stack().reset_index()
f.setupplot(figsize=(7,7))
fig, ax = plt.subplots(1,2, sharey=True)
sns.barplot(data=plotdf, y='level_1', x=0, hue='Condition.MAM', orient='h', ax=ax[0])
sns.barplot(data=prevdf, y='level_0', x=0, hue='level_1',ax=ax[1])
f.savefig('top50boxdiff')

# Maaslin volcano
maaslin = f.load('taxochange')
fmas = f.filter(maaslin, query='metadata == "Condition.MAM"', rowfilt='s__')
fmas.index = fmas.index.str.replace('s__','').str.replace('_',' ')
df = f.load('taxo') # reload species
df = df.loc[:, df.columns.str.startswith('s')]
df.columns = df.columns.str.replace('s__','').str.replace('_',' ')
df = f.stratify(df, meta, 'Condition.MAM')
prevs = prev(df)
jmas = fmas.join(prevs)
jmas['basemean'] = jmas['basemean'].apply(np.sqrt).mul(30)
f.setupplot(figsize=(2.2,2.2), fontsize=5)
f.volcano(jmas, change='coef',sig='pval',fc=0.1,pval=0.05,size='basemean', )
legend_sizes = [np.sqrt(10)*30, np.sqrt(1)*30, np.sqrt(0.1)*30]
scatter_for_legend = [plt.scatter([], [], c='black',s=size, alpha=0.5) for size in legend_sizes]
plt.legend(scatter_for_legend, [f'{size}' for size in legend_sizes], title="Mean Relative Abundance")
f.savefig('sizevolcano')

'''
clr_data = f.CLR(condition)
change = pd.concat([change, f.mww(clr_data, comb=['Malnourished', 'Well-nourished'])], axis=1)
pval = 0.05
sig = f.filter(change, column='MWW_pval', lt=pval)
changesig = change.loc[sig.index]
sig['deltaprevail'] = sig.Malnourishedprev - sig['Well-nourishedprev']

fig, ax = plt.subplots(1,2, figsize=(3,5), sharey=True)
sig['deltaprevail'].sort_values().mul(100).plot.barh(ax=ax[0])
ax[0].grid()
sig['Log2FC'].sort_values().plot.barh(ax=ax[1])
ax[1].grid()
f.savefig('specieschange')
f.save(change, 'specieschange')
'''
