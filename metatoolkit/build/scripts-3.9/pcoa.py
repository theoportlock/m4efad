#!/home/theop/venv/bin/python3
# -*- coding: utf-8 -*-

import argparse
import functions as f
import pandas as pd
from pathlib import Path
import os
import skbio

parser = argparse.ArgumentParser(description='Filter')
parser.add_argument('subject', type=str, help='Distance matrix')
parser.add_argument('-o', '--outfile', type=str)
parser.add_argument('-s', '--suffix', type=str)
known = parser.parse_args()
known = {k: v for k, v in vars(known).items() if v is not None}

subject = known.get("subject"); known.pop('subject')
if os.path.isfile(subject): subject = Path(subject).stem
df = f.load(subject)

outfile = known.get("outfile") if known.get("outfile") else None
suffix = known.get("suffix") if known.get("suffix") else None

output = pd.DataFrame(index=df.index)
DM_dist = skbio.stats.distance.DistanceMatrix(df)
PCoA = skbio.stats.ordination.pcoa(DM_dist, number_of_dimensions=2)
label = PCoA.proportion_explained.apply(' ({:.1%})'.format)
results = PCoA.samples.copy()
output['PCo1' + label.loc['PC1']], output['PCo2' + label.loc['PC2']] = results.iloc[:,0].values, results.iloc[:,1].values

print(output)
if output is not None:
    if outfile:
        f.save(output, outfile)
    elif suffix:
        f.save(output, subject + outfile)
    else:
        f.save(output, subject + 'Pcoa')
