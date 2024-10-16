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
alldf = f.load('alldata')
meta = f.load('meta')

# Connections
salldf = alldf.join(meta)
f.setupplot(figsize=(1,1), fontsize=5)

y='s__Bacteroides_fragilis'
f.box(salldf, y=y, x='Condition')
f.savefig('bacterbox')

y='s__Streptococcus_salivarius'
f.box(salldf, y=y, x='Condition')
f.savefig('strepbox')

y='PWY-6876: isopropanol biosynthesis (engineered)'
f.box(salldf, y=y, x='Condition')
f.savefig('isobox')

y='GLUCOSE1PMETAB-PWY: glucose and glucose-1-phosphate degradation'
f.box(salldf, y=y, x='Condition')
f.savefig('glucobox')

y='Expressive Communication Score'
f.box(salldf, y=y, x='Condition')
f.savefig('ecscorebox')

x='PWY-6876: isopropanol biosynthesis (engineered)'
y='Gross Motor Score'
sns.regplot(data = salldf, x=x, y=y, color='red', scatter=False)
sns.scatterplot(data = salldf, x=x, y=y, s=2, hue='Condition', legend=False)
f.savefig('isoECcorr')

y='s__Bacteroides_fragilis'
x='s__Streptococcus_salivarius'
salldf[x] = (salldf[x] + 0.0001).apply(np.log)
salldf[y] = (salldf[y] + 0.0001).apply(np.log)
sns.regplot(data = salldf, x=x, y=y, color='blue', scatter=False)
sns.scatterplot(data = salldf, x=x, y=y, s=2, hue='Condition', legend=False)
f.savefig('strepbactercorr')

x='s__Bacteroides_fragilis'
y='P108-PWY: pyruvate fermentation to propanoate I'
sns.regplot(data = salldf, x=x, y=y, color='red', scatter=False)
sns.scatterplot(data = salldf, x=x, y=y, s=2, hue='Condition', legend=False)
f.savefig('bacterpyruvatecorr')

y='Vocalisation'
x='GLUCOSE1PMETAB-PWY: glucose and glucose-1-phosphate degradation'
sns.regplot(data = salldf, x=x, y=y, color='red', scatter=False)
sns.scatterplot(data = salldf, x=x, y=y, s=2, hue='Condition', legend=False)
f.savefig('vocglucosecorr')
