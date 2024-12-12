#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import functions as f
import pandas as pd
import os
from pathlib import Path

parser = argparse.ArgumentParser(description='Describe - Produces a summary report of analysis')
parser.add_argument('subject')
parser.add_argument('-p', '--pval', type=float, default=0.25)
parser.add_argument('-c', '--change', type=str, default='coef')
parser.add_argument('-s', '--sig', type=str, default='qval')
known = parser.parse_args()
known = {k: v for k, v in vars(known).items() if v is not None}

def change_summary(df, change='coef', sig='qval', pval=0.25):
    total_rows = df.shape[0]
    sig_changed_count = df[sig].lt(pval).sum()
    changed = f"sig changed = {sig_changed_count}/{total_rows} ({round(sig_changed_count / total_rows * 100)}%)"
    sig_increased_count = df.loc[(df[sig] < pval) & (df[change] > 0), sig].lt(pval).sum()
    increased = f"sig up = {sig_increased_count}/{total_rows} ({round(sig_increased_count / total_rows * 100)}%)"
    sig_decreased_count = df.loc[(df[sig] < pval) & (df[change] < 0), sig].lt(pval).sum()
    decreased = f"sig down = {sig_decreased_count}/{total_rows} ({round(sig_decreased_count / total_rows * 100)}%)"
    return pd.Series([changed, increased, decreased])

subject = known.get("subject")
if os.path.isfile(subject): subject = Path(subject).stem
df = f.load(subject)
pval = known.get("pval")
change = known.get("change")
sig = known.get("sig")

output = change_summary(df, change=change, pval=pval, sig=sig)
print(output.to_string())
f.save(output, f'{subject}Summary')
