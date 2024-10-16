#!/usr/bin/env python
import pandas as pd
import metatoolkit.functions as f
import matplotlib.pyplot as plt

df = f.load('shaps')
df = df.drop(['metadata','value','N','N.not.0','pval','qval'], axis=1)
meandf= df.groupby('dataset').sum()

f.setupplot()
fig, ax = plt.subplots(figsize=(4, 4))  # Set the figure size
ax.pie(
    meandf['SHAPmean'],  # Data for the pie chart
    labels=meandf.index,  # Labels for each segment
    autopct='%1.1f%%',  # Display percentage on segments
    colors=plt.cm.Paired.colors,  # Use a predefined color map
    startangle=90,  # Start the first segment at 90 degrees
    wedgeprops={'edgecolor': 'black', 'linewidth': 1}  # Add black edge around segments
)
ax.set_title('SHAP Mean by Dataset', fontsize=16)  # Title with increased font size
f.savefig('shappie')
