#!/usr/bin/env python
import pandas as pd
import numpy as np
import functions as f
from statsmodels.stats.multitest import fdrcorrection

# load data
fisher = f.load('categoriesfisher')
mww = f.load('numericchange')
cor = f.load('numericcorr')

# Apply log of odds to get negative association and tanh to make -1 to 1
fisher['effect'] = fisher.odds.apply(np.log2).apply(np.tanh)
fisher['pval'] = fisher['pval']
fisher['test'] = 'fisher'
fisher = fisher.reset_index().set_index(['source','target'])

# Have to use diffmean here to preserve effect direction for negative numbers
mww['effect'] = mww['diffmean']
mww['pval'] = mww['MWW_pval']
mww['test'] = 'mww'
mww = mww.reset_index().set_index(['source','target'])

cor['effect'] = cor['cor']
cor['pval'] = cor['pval']
cor['test'] = 'spear'
cor = cor.reset_index().set_index(['source','target'])

# Merge
outdf = pd.concat([cor[['effect', 'pval', 'test']],
                   mww[['effect', 'pval', 'test', 'source_true_summary', 'source_false_summary']],
                   fisher[['effect', 'pval', 'test', 'source_true_summary','source_false_summary']]],
                  axis=0)

# Drop NA
outdf = outdf.dropna()

# Adjust p-values to q-values
outdf['qval'] = fdrcorrection(outdf['pval'])[1]

# save
f.save(outdf, 'covariateedges')

mww['effect'] = mww['Log2FC'].apply(np.tanh)

