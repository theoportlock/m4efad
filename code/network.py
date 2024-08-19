#!/usr/bin/env python
from itertools import permutations
from scipy.stats import spearmanr
from statsmodels.stats.multitest import fdrcorrection
import metatoolkit.functions as f
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Load data
alldata = f.load('alldata')
alldata.columns = alldata.columns.str.replace(r'[\[\]\;\/\|\(\)\:\-\ ]','.', regex=True)

# significant changes
shaps = f.load('shaps')
print(shaps.SHAPmean.quantile(0.85))
sigvals = shaps.loc[shaps.SHAPmean.gt(shaps.SHAPmean.quantile(0.85))].index
fshaps = shaps.loc[sigvals]

# filter sig
alldf = alldata.loc[:, fshaps.index]

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

# Add shap interacts
si = f.load('shap_interactsmeanformatfilter').reset_index().set_index(['source', 'target'])
foutdf = foutdf.reset_index().set_index(['source', 'target'])
joined = foutdf.join(si, how='inner')

#f.save(foutdf, 'correlations')
#joined['relationship'] = np.sign(joined.cor)
f.save(joined, 'correlations')
