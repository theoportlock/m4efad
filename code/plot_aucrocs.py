#!/usr/bin/env python
import matplotlib.pyplot as plt
import metatoolkit.functions as f
import numpy as np
import pandas as pd
import seaborn as sns
import shap

# load and format data
df = f.load('allaucrocs').reset_index()
df.columns = df.columns.str.replace('Condition.MAM','')
df = df.loc[:, df.mean().sort_values(ascending=False).index]

# Plot
f.setupplot(fontsize=6.5)
fig, ax = plt.subplots(figsize=(2.5, 2.5))
sns.barplot(df, orient='y', ax=ax)
ax.set_xlabel('AUCROC')
ax.set_ylabel('Dataset')
f.savefig('aucrocvals')

# Print
print(df.agg(['mean','std']))
