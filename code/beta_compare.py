#!/usr/bin/env python
from glob import glob
from itertools import permutations
from pathlib import Path
from skbio import DistanceMatrix
from skbio.stats.distance import mantel
import matplotlib.pyplot as plt
import metatoolkit.functions as f
import numpy as np
import pandas as pd
import sys
import seaborn as sns

# Load data
#files = glob('../results/beta*')
files = ['../results/beta_bray-curtis.tsv',
         '../results/beta_jaccard.tsv',
         '../results/beta_weighted-unifrac.tsv',
         '../results/beta_unweighted-unifrac.tsv']
basefiles = [Path(file).stem for file in files]

# Read and join the input matrices
matrices = [f.load(matrix).stack().to_frame(matrix) for matrix in basefiles]
corr = pd.concat(matrices, axis=1, join='inner')

# Plot
f.setupplot()
sns.pairplot(corr,
             kind="scatter",
             plot_kws={'s':2},
             diag_kind='hist',
             diag_kws={'bins':20, 'fill':False},
             corner=True)
f.savefig('mantel')

# Mantel
# Reload data
matrices = [f.load(matrix) for matrix in basefiles]
dm = pd.Series([DistanceMatrix(mat.values, ids=mat.index) for mat in matrices], index=basefiles)

# Perform test for all combinations
names = list(permutations(basefiles, 2))
outdf = pd.DataFrame(index=pd.MultiIndex.from_tuples(names), columns=['cor','pval'])
for name in names:
    print(name)
    outdf.loc[name] = mantel(dm[name[0]], dm[name[1]], permutations=999)[:2]

# Add FDR correction
outdf['qval'] = f.fdrcorrection(outdf.pval)[1]
outdf = outdf.drop(['pval'], axis=1)

# Save results
print(outdf)
f.setupplot()
f.clustermap(outdf['cor'].unstack().fillna(0))
f.savefig('mantelCM')
