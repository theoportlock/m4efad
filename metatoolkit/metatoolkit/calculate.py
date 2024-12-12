#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import functions as f
import pandas as pd
from pathlib import Path
import os

parser = argparse.ArgumentParser(description='''
Calculate - compute a value for each sample based on features
''')

parser.add_argument('analysis', type=str, help='Name of data')
parser.add_argument('subject', type=str, help='Type of analysis')
parser.add_argument('--axis', type=str, help='Data axis for computation')
known = parser.parse_args()
known = {k: v for k, v in vars(known).items() if v is not None}

subject = known.get("subject")
axis = known.get("axis") if known.get("axis") else 'rows'
if os.path.isfile(subject): subject = Path(subject).stem
df = f.load(subject)

analysis = known.get('analysis')
if axis == 'rows':
    output = f.calculate(analysis, df)
elif axis == 'columns':
    df = df.T
    output = f.calculate(analysis, df)
    output = output.T

print(output)
f.save(output, subject + known.get("analysis"))
