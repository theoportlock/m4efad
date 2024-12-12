#!/home/theop/venv/bin/python3
# -*- coding: utf-8 -*-

import argparse
import functions as f
import matplotlib.pyplot as plt
import pandas as pd
import os

parser = argparse.ArgumentParser(description='''
Box - Produces a Boxplot of a given dataset
''')

parser.add_argument('subject')
parser.add_argument('-x')
parser.add_argument('-y')
parser.add_argument('--hue')
parser.add_argument('--logy', action='store_true')

known = parser.parse_args()
known = {k: v for k, v in vars(known).items() if v is not None}

subject = known.get("subject"); known.pop("subject")
if os.path.isfile(subject): subject = Path(subject).stem
df = f.load(subject)
logy = known.get("logy"); known.pop("logy")

f.setupplot()
f.box(df, **known)
if logy: plt.yscale('log')
f.savefig(f'{subject}box')

