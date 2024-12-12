#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import functions as f
import pandas as pd
import os

parser = argparse.ArgumentParser(description='''
Change - Bivariate analysis of the feature changes
''')

parser.add_argument('subject', type=str)
parser.add_argument('-a', '--analysis', nargs='+', help='Methods of analysis by which to measure the difference')
parser.add_argument('-c', '--columns', nargs='+', help='Which columns of second dataframe to measure differences')
parser.add_argument('-df2', '--df2', default='meta', help='Directory of table with one hot encoded variables that contain the labels used to compare differences')
known = parser.parse_args()
known = {k: v for k, v in vars(known).items() if v is not None}

subject = known.get("subject")
columns = known.get("columns") if known.get("columns") else None
df2 = known.get("df2") if known.get("df2") else 'meta'

if os.path.isfile(subject): subject = Path(subject).stem
df = f.load(subject)

if os.path.isfile(df2): subject = Path(df2).stem
df2 = f.load(df2)

output = f.change(df, df2, columns=columns)
print(output)

f.save(output, subject+'change')

