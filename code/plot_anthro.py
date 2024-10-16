#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
import metatoolkit.functions as f

# Load data
df = f.load('anthro')
df = df.rename(columns={'Weight':'Weight (Kg)', 'MUAC':'MUAC (cm)', 'HC':'HC (cm)', 'WLZ_WHZ':'WLZ/WHZ'}) 
meta = f.load('meta')

# Stratify anthropometrics
strat = f.stratify(df, meta, 'Condition')

# Plot
f.setupplot(figsize=(1.8, 1.8), fontsize=5)

# Assuming f.box takes both ax and the column dataframe as inputs
fig, ax = plt.subplots(2, 2, figsize=(1.8, 1.8))  # Creating a 2x2 grid for subplots
fig.tight_layout(pad=0.01)  # Adjust spacing

# Flatten the 2x2 grid of axes to simplify iteration
axes = ax.flatten()

# Iterate over the columns and axes
for i, anthrocol in enumerate(strat.columns):
    f.box(strat[anthrocol].to_frame(), ax=axes[i])

# Save
f.savefig('anthrobox')
