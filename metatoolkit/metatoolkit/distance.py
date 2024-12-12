#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import metatoolkit.functions as f
import pandas as pd
from pathlib import Path
import os
from skbio import stats
from scipy.spatial.distance import pdist, squareform
from skbio.stats.distance import DistanceMatrix

parser = argparse.ArgumentParser(description='Filter')
parser.add_argument('subject', type=str, help='Distance matrix')
parser.add_argument('-m', '--metric', type=str, help='Distance metric - could be braycurtis, canberra, chebyshev, cityblock, correlation, cosine, dice, euclidean, hamming, jaccard, jensenshannon, kulczynski1, mahalanobis, matching, minkowski, rogerstanimoto, russellrao, seuclidean, sokalmichener, sokalsneath, sqeuclidean, or yule')
parser.add_argument('-o', '--outfile', type=str)
parser.add_argument('-s', '--suffix', type=str)
known = parser.parse_args()
known = {k: v for k, v in vars(known).items() if v is not None}

# Load variables
subject = known.get("subject")
if os.path.isfile(subject): subject = Path(subject).stem
df = f.load(subject)
outfile = known.get("outfile") if known.get("outfile") else None
suffix = known.get("suffix") if known.get("suffix") else None
metric = known.get("metric") if known.get("metric") else 'braycurtis'

# Calculate
output = f.dist(df, metric=metric)

# Save
print(output)
if output is not None:
    if outfile:
        f.save(output, outfile)
    elif suffix:
        f.save(output, subject + outfile)
    else:
        f.save(output, subject + 'Dist')
