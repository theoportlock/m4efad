#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import functions as f
import pandas as pd
from pathlib import Path
import os

parser = argparse.ArgumentParser(description='Filter')
parser.add_argument('subject')
parser.add_argument('-rf', '--rowfilt', type=str, help='regex for index filtering')
parser.add_argument('-cf', '--colfilt', type=str, help='regex for column filtering')
parser.add_argument('-q', '--query', type=str, help='Pandas custom query')
parser.add_argument('-m', '--min_unique', type=int, help='Minimum number')
parser.add_argument('-fdf', '--filter_df')
parser.add_argument('-fdfx', '--filter_df_axis', type=int)
parser.add_argument('-absgt', type=float)
parser.add_argument('-p', '--prevail', type=float)
parser.add_argument('-a', '--abund', type=float)
parser.add_argument('-o', '--outfile', type=str)
parser.add_argument('-s', '--suffix', type=str)
parser.add_argument('--numeric_only', action='store_true')
parser.add_argument('--nonzero', action='store_true')
parser.add_argument('-dt', '--dtype', type=str, help='select columns with dtype')
known = parser.parse_args()
known = {k: v for k, v in vars(known).items() if v is not None}

subject = known.get("subject"); known.pop('subject')
if os.path.isfile(subject): subject = Path(subject).stem
df = f.load(subject)

outfile = known.get("outfile") if known.get("outfile") else None
suffix = known.get("suffix") if known.get("suffix") else None

if known.get("filter_df"):
    known['filter_df'] = f.load(known.get("filter_df"))

output = f.filter(df, **known)
print(output)
if output is not None:
    if outfile:
        f.save(output, outfile)
    elif suffix:
        f.save(output, subject + outfile)
    else:
        f.save(output, subject + 'filter')
