#!/usr/bin/env python
import metatoolkit.functions as f
import pandas as pd
from glob import glob

# Load files and get value counts for conditions
files = glob('../results/*Condition.MAM.tsv')
dfs = [f.load(file).index.value_counts().to_frame(file) for file in files]

outdf = pd.concat(dfs, axis=1).T
outdf.loc[:, 'all'] = outdf.sum(axis=1)

print('Sample numbers:')
print(outdf)


