#!/usr/bin/env python
import pandas as pd
from skbio.stats.composition import multiplicative_replacement as mul
import numpy as np
import functions as f
import matplotlib.pyplot as plt

# load data
df = f.load('quality')
meta = f.load('meta')
anthro = f.load('anthro')

# Quality
condition = f.stratify(df, meta, 'Condition')
print(f.change(condition, analysis=['prevail', 'mww'])['MalnourishedvsWell-nourished'].T)
print(condition.mean())
print(condition.std())

# Diversity
df = f.load('species')
diversity = f.calculate('diversity', df)
diversityCondition = f.stratify(diversity, meta, 'Condition')
print(diversity.mean(), diversity.std())
diversityConditionchange = f.change(diversityCondition)['MalnourishedvsWell-nourished']
f.save(diversityConditionchange, 'diversitychange')
f.setupplot(figsize=(1,3))
#f.multibox(diversityCondition[['Shannon','Evenness']], sharey=False)
f.box(diversityCondition['Shannon'].to_frame())
f.savefig('speciesdiversityConditionbox')

# PCOA
condition = f.stratify(df, meta, 'Condition')
pcoa = f.calculate('pcoa', condition)
ax = f.plot('spindle', pcoa)
ax.set_xlabel('PCoA1 (19.2%)')
ax.set_ylabel('PCoA1 (0.09%)')
print(f.PERMANOVA(condition, pval=True))
print(f.PERMANOVA(condition, pval=False))
f.savefig('speciesConditionpcoa')

# taxa changes
condition = f.stratify(df, meta, 'Condition')
condition = f.filter(condition, prevail=0.1)
change = f.change(condition, analysis=['prevail','fc'])['MalnourishedvsWell-nourished']
condition = pd.DataFrame(mul(condition, delta=1e-6), index=condition.index, columns=condition.columns)
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

#individ plot
up = sig.loc[sig['Well-nourishedmean'].gt(sig['Malnourishedmean'])]
down = sig.loc[sig['Well-nourishedmean'].lt(sig['Malnourishedmean'])]
f.describe(clrchange, change='Log2FC', sig='MWW_pval')
f.save(clrchange, 'specieschange')
sigfilt = f.filter(condition, filter_df=sig, filter_df_axis=1)
#f.multibox(sigfilt, logy=True)
upfilt = f.filter(condition, filter_df=up, filter_df_axis=1)
downfilt = f.filter(condition, filter_df=down, filter_df_axis=1)
f.setupplot(figsize=(1,2))
#f.multibox(downfilt, logy=True)
f.box(downfilt['Prevotella_copri'].to_frame())
plt.yscale('log')
f.savefig('specdownbox')
f.setupplot(figsize=(1,2))
f.box(upfilt)
plt.yscale('log')
f.savefig('specupbox')

# ratio changes
genus = f.load('genus')
genus = f.filter(genus, nonzero=True)
pbratio = f.calculate("pbratio", f.mult(genus))
pbratcondition = f.stratify(pbratio, meta, 'Condition')
pbratconditionchange = f.change(pbratcondition)['MalnourishedvsWell-nourished']
f.setupplot(figsize=(2,2))
f.box(pbratcondition)
plt.yscale('log')
f.savefig('pbratiobox')
print(pbratconditionchange.T)

# Functions
df = f.load('pathwaysall')
condition = f.stratify(df, meta, 'Condition')
condition = f.filter(condition, prevail=0.2, abund=0.00001)
change = f.change(condition)['MalnourishedvsWell-nourished']
print(f.describe(change, change='Log2FC', sig='MWW_pval'))
print(f.describe(change, change='Log2FC', sig='MWW_pval', pval=0.1))
f.save(change, 'pathwayschange')
f.setupplot(figsize=(2.5,2.5))
f.volcano(change, fc=0.1, pval=0.1, annot=True)
f.savefig('funcvolcano')
sig = f.filter(change, column='MWW_pval', lt=0.1)

# Just anaerobic pathways
fdf = df.loc[:, df.columns.str.contains('fermentation')].sum(axis=1).to_frame('fermentation')
sfdf = f.stratify(fdf, meta, 'Condition')
ch = f.change(sfdf)['MalnourishedvsWell-nourished'].T
print(ch)
f.box(sfdf)

# species contribution
df = f.load('pathways')
df = f.stratify(df, meta, 'Condition')

ile = df[df.columns[df.columns.str.contains('superpathway of branched chain amino acid')]].iloc[:,1:]
#ile = df[df.columns[df.columns.str.contains('ANAGLYCOLYSIS')]].iloc[:,1:]
ile.columns = ile.columns.str.replace('.*\|', '', regex=True)
ile.columns = ile.columns.str.replace('.*s__', '', regex=True)
means = ile.groupby(level=0).mean()
f.setupplot()
f.abund(means)
f.savefig('BCAAspecies')
ch = f.change(ile)['MalnourishedvsWell-nourished']
sig = ch.loc[ch.MWW_pval.lt(0.05)]
f.volcano(ch, fc=0, pval=0.1)
