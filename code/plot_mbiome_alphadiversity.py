import pandas as pd
import metatoolkit.functions as f
import matplotlib.pyplot as plt

df = f.load('alpha_diversitychange').reset_index()

df = df.set_index(['source','target'])

sig = df.MWW_qval.unstack()

lfc = df.Log10FC.unstack()

fsig = sig.loc[sig.lt(0.2).any(axis=1)]

flfc = lfc.loc[fsig.index]

f.setupplot()

f.clustermap(flfc, fsig.lt(0.2), figsize=(4,4))

f.savefig('alphacluster')

