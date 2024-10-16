#!/usr/bin/env python
import pandas as pd
import metatoolkit.functions as f
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

shaps = f.load('shaps')['SHAPmean']
inter = f.load('shap_interactsmeanformat').reset_index().set_index(['source','target'])['shap_interactsmean'].abs()

def plothist(df, thresh=90):
    quantile_90 = np.quantile(df, 0.90)
    print('90quantile = ', quantile_90)
    print('meet = ', df.gt(quantile_90).value_counts())
    counts, bins, patches = plt.hist(df, bins=30, edgecolor='black')
    for patch, left_edge in zip(patches, bins[:-1]):
        if left_edge >= quantile_90:
            patch.set_facecolor('red')  # Color for bars over the 90 quantile
        else:
            patch.set_facecolor('gray')    # Color for bars below the 90 quantile
    plt.xlabel('Value')
    plt.ylabel('Frequency')

f.setupplot(figsize=(6,6))

print('shap')
plothist(shaps.drop_duplicates())
plt.yscale('log')
plt.gca().yaxis.set_major_formatter(ScalarFormatter())
plt.grid()
f.savefig('shaphist')

print('shapinteract')
df = inter.drop_duplicates()
df_filtered = df[df.index.get_level_values(0) != df.index.get_level_values(1)]
plothist(df_filtered)
plt.yscale('log')
plt.gca().yaxis.set_major_formatter(ScalarFormatter())
plt.grid()
f.savefig('shapinterhist')
