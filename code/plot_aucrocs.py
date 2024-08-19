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

# plot
f.setupplot(figsize=(4,4))
ax = sns.barplot(df, orient='y')
ax.set_xlabel('AUCROC')
ax.set_ylabel('Dataset')
plt.tight_layout()
f.savefig('aucrocvals')

