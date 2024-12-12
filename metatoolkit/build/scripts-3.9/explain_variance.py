#!/home/theop/venv/bin/python3
from scipy.spatial import distance
from skbio.stats.distance import permanova, permdisp, DistanceMatrix
import argparse
import functions as f
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

parser = argparse.ArgumentParser(description='''
Explained Variance - takes dataset(s) and runs PERMANOVA on the beta diversities
''')
parser.add_argument('dfs', type=str, nargs='+', help='dataset(s) for PERMANOVA calculation')
parser.add_argument('--df_cats', default='meta', type=str, help='metadata with categories only')
parser.add_argument('-o', '--outfile', type=str)
parser.add_argument('-s', '--suffix', type=str)
parser.add_argument('--from_beta', action='store_true')
known = parser.parse_args()
known = {k: v for k, v in vars(known).items() if v is not None}

# Load data
dfs = {df:f.load(df) for df in known.get("dfs")}
outfile = known.get("outfile") if known.get("outfile") else None
from_beta = known.get("from_beta") if known.get("from_beta") else None
suffix = known.get("suffix") if known.get("suffix") else None
df_cats = f.load(known.get('df_cats'))

# Find combinations
combinations = [(x, y) for x in dfs.keys() for y in df_cats.columns]

# Calculate PERMANOVA power
np.random.seed(0)
output = pd.DataFrame(index=combinations, columns=['R2', 'anov_pval', 'dispstat', 'disp_pval'])
for data_key, column in combinations:
    data = dfs[data_key]
    if from_beta:
        dist = DistanceMatrix(data)
    else:
        beta = distance.squareform(distance.pdist(data, metric="braycurtis"))
        dist = DistanceMatrix(beta)
    anov = permanova(dist, df_cats, column=column)
    disp = permdisp(dist, df_cats, column=column)
    output.loc[(data, column), ['R2', 'anov_pval']] = anov['test_statistic'], anov['p-value']
    output.loc[(data, column), ['dispstat', 'disp_pval']] = disp['test_statistic'], disp['p-value']

# Save output
print(output)
if output is not None:
    if outfile:
        f.save(output, outfile)
    elif suffix:
        f.save(output, subject + outfile)
    else:
        f.save(output, subject + 'perm')
