#!/usr/bin/env python
import pandas as pd
import numpy as np
import functions as f
import matplotlib.pyplot as plt

# Load
meta = f.load('meta')
lipids = f.load('lipids')
metab = f.load('metab')

# lipid changes
metabcondition = f.stratify(metab, meta, 'Condition')
metabconditionchange = f.change(metabcondition)['MalnourishedvsWell-nourished']
f.describe(metabconditionchange, change="Log2FC", sig="MWW_qval")
f.setupplot()
f.volcano(metabconditionchange, annot=True, fc=0.5, pval=0.0005)
plt.xlim([-2.5, 2.5])
f.savefig('lipidvolc')
metabconditionchange = metabconditionchange.loc[:, ~metabconditionchange.columns.str.contains('prev')]
f.save(metabconditionchange, 'lipidchange')

# grouped by lipid class
metabcondition = f.stratify(lipids, meta, 'Condition')
metabconditionchange = f.change(metabcondition)['MalnourishedvsWell-nourished']
metabconditionchangefilt = f.filter(metabconditionchange, column='MWW_qval', lt=0.1)
f.setupplot(figsize=(6,2))
f.multibox(metabcondition.loc[:,metabconditionchangefilt.index], sharey=False)
f.savefig('lipidclasschangebox')
metabconditionchange = metabconditionchange.loc[:, ~metabconditionchange.columns.str.contains('prev')]
f.save(metabconditionchange, 'lipidclasschange')

# Lysolipid changes
metab = f.load('metab')
lysolipids = f.filter(metab, colfilt='^L.*').sum(axis=1).to_frame('lysolipids')
lysolipidscondition = f.stratify(lysolipids, meta, 'Condition')
lch = f.change(lysolipidscondition)['MalnourishedvsWell-nourished']
f.setupplot(figsize=(1.5,1.5))
f.box(lysolipidscondition)
f.savefig('lysolipidsbox')

# OCFA changes
metab = f.load('metab')
fa = metab.loc[:,metab.columns.str.startswith('FA ')]
fa.columns = fa.columns.str.replace('FA ', '')
fa.columns = fa.columns.str.replace(':.*', '', regex=True)
fa.columns = fa.columns.astype(int)
ecfa = fa.loc[: , fa.columns[fa.columns % 2 == 0]]
ecfa = ecfa.sum(axis=1).to_frame('ECFA')
ecfacondition = f.stratify(ecfa, meta, 'Condition')
ocfa = fa.loc[: , fa.columns[fa.columns % 2 != 0]]
ocfa = ocfa.sum(axis=1).to_frame('OCFA')
f.setupplot(figsize=(1.5,1.5))
ocfacondition = f.stratify(ocfa, meta, 'Condition')
f.box(ocfacondition)
f.savefig('ocfabox')
och = f.change(ocfacondition)['MalnourishedvsWell-nourished']

# Save OCFA and Lysolipids
extralipids = ocfa.join(lysolipids)
f.save(extralipids, 'extralipids')
extrachange = pd.concat([och, lch])
f.save(extrachange, 'extralipidschange')
