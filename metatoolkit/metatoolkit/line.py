#!/usr/bin/env python
# -*- coding: utf-8 -*-

import seaborn as sns
import argparse
import functions as f
import matplotlib.pyplot as plt
import pandas as pd
import os

parser = argparse.ArgumentParser(description='''
Line - Produces a Lineplot of a given dataset
''')
parser.add_argument('subject')
parser.add_argument('-df2', '--df2', type=str, default='meta', help='categorical data to label lineplot with')
parser.add_argument('-x')
parser.add_argument('-y')
parser.add_argument('--hue')
parser.add_argument('--units')
parser.add_argument('--logy', action='store_true')
known = parser.parse_args()
known = {k: v for k, v in vars(known).items() if v is not None}

# Load data
subject = known.get('subject')
if os.path.isfile(subject): subject = Path(subject).stem
df = f.load(subject)
df2 = f.load(known.get('df2'))

# Merge metadata
plotdf = df.join(df2)
print(plotdf)

# Load variables
units = known.get('units')
x = known.get("x")
y = known.get("y")
hue = known.get("hue")
logy = known.get("logy")

# Sort
plotdf = plotdf.sort_values(x)

# Plot and save
f.setupplot()
sns.lineplot(data=plotdf,
             x=x,
             y=y,
             units=units,
             hue=hue,
             estimator=None)
if logy: plt.yscale('log')
f.savefig(f'{subject}line')

