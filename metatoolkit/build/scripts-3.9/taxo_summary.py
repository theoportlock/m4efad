#!/home/theop/venv/bin/python3
# -*- coding: utf-8 -*-

import argparse
import functions as f
import pandas as pd
import os
from pathlib import Path

parser = argparse.ArgumentParser(description='Describe - Produces a summary report of taxonomic breakdown')
parser.add_argument('subject')
known = parser.parse_args()
known = {k: v for k, v in vars(known).items() if v is not None}

def taxo_summary(df):
    count = df.columns.str.replace(".*\|", "", regex=True).str[0].value_counts().to_frame('all_count')
    # Individual stats
    output = []
    for sample in df.index:
        samp = df.loc[sample]
        fsamp = samp[samp != 0]
        output.append(fsamp.index.str.replace(".*\|", "", regex=True).str[0].value_counts())
    outdf = pd.concat(output, axis=1).set_axis(df.index, axis=1).T
    outdf.mean()
    odf = pd.concat([count, outdf.mean().to_frame('mean_count'), outdf.std().to_frame('std_count')], axis=1)
    return odf

subject = known.get("subject")
if os.path.isfile(subject): subject = Path(subject).stem
df = f.load(subject)

output = taxo_summary(df)
print(output.to_string())
f.save(output, f'{subject}Taxocount')
