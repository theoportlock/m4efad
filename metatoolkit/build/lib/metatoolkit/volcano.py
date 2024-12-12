#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import functions as f
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path

parser = argparse.ArgumentParser(description='''
Volcano - Produces a Volcano plot of a given dataset
''')

parser.add_argument('subject', type=str)
parser.add_argument('--change', type=str)
parser.add_argument('--sig', type=str)
parser.add_argument('--fc', type=float)
parser.add_argument('--pval', type=float)
parser.add_argument('--annot', action=argparse.BooleanOptionalAction)

known, unknown = parser.parse_known_args()
known = {k: v for k, v in vars(known).items() if v is not None}
unknown = eval(unknown[0]) if unknown != [] else {}

# Assemble params
subject = known.get("subject"); known.pop('subject')
change = known.get("change") if known.get("change") else 'Log2FC'
sig = known.get("sig") if known.get("sig") else 'MWW_pval'
fc = float(known.get("fc")) if known.get("fc") else 1.0
pval = float(known.get("pval")) if known.get("pval") else 0.05
annot = known.get("annot") if known.get("annot") else True

# Load data
df = f.load(subject)

# Plot
f.setupplot()
f.volcano(df, change=change, sig=sig, fc=fc, pval=pval, annot=annot)
if os.path.isfile(subject): subject = Path(subject).stem
f.savefig(f'{subject}volcano')
