#!/usr/bin/env python
import metatoolkit.functions as f
from skbio.stats.composition import multi_replace
import numpy as np

# Load data
meta = f.load('categories')
df = f.load('taxo')

# Format
df.columns = df.columns.str.replace('.*\|','', regex=True)

# Filter just P and B
df.columns = df.columns.str[3:]
df = df.T.apply(multi_replace, axis=0).T
df = df.apply(np.log)

#pb = df.Prevotella / (df.Prevotella + df.Bacteroides)
pb = df.Prevotella / df.Bacteroides
pb = pb.to_frame('PBratio')

# Save 
f.save(pb, 'PBratio')

# Measure changes
ch = f.change(pb, meta)
fch = f.filter(ch.reset_index().set_index('source'), query='MWW_qval < 0.05')
print(fch.loc['Condition.MAM'])


