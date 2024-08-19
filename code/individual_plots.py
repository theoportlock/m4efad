#!/usr/bin/env python
from itertools import permutations
from scipy.stats import spearmanr
from statsmodels.stats.multitest import fdrcorrection
import functions as f
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Load data
alldf = f.load('alldata')
meta = f.load('meta')

# Connections
salldf = f.stratify(alldf, meta, 'Condition')
f.setupplot(figsize=(1,1))
y='s__Bacteroides_fragilis'
f.box(salldf, y=y)
f.savefig(y)

y='P108-PWY: pyruvate fermentation to propanoate I'
f.box(salldf, y=y)
f.savefig('pyruvatediffs')

y='Expressive Communication Score'
f.box(salldf, y=y)
f.savefig(y)

x='P108-PWY: pyruvate fermentation to propanoate I'
y='Expressive Communication Score'
sns.regplot(data = salldf, x=x, y=y, scatter_kws={"color": "black", 's':1}, line_kws={"color": "red"})
f.savefig('pyruvatediffs'+y)

x='Vocalisation'
y='GLYCOL-GLYOXDEG-PWY: superpathway of glycol metabolism and degradation'
sns.regplot(data = salldf, x=x, y=y, scatter_kws={"color": "black", 's':1}, line_kws={"color": "red"})
f.savefig('pyruvatediffsglycol')
