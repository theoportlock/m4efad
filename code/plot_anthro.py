#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
import functions as f

# Load data
df = f.load('anthro')
df = df.rename(columns={'Weight':'Weight (Kg)', 'MUAC':'MUAC (cm)', 'HC':'HC (cm)', 'WLZ_WHZ':'WLZ/WHZ'}) 
meta = f.load('meta')

# Stratify anthropometrics
strat = f.stratify(df, meta, 'Condition')

# Plot
f.setupplot()
f.multiviolin(strat.sort_index(ascending=False), sharey=False)
f.savefig('anthrovioin')
