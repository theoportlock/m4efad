#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import functions as f
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser(description='''
Regplot - Produces a Regplot of a given dataset
''')

parser.add_argument('subject')
parser.add_argument('-x')
parser.add_argument('-y')
parser.add_argument('--hue')
parser.add_argument('--logy', action='store_true')

known = parser.parse_args()
known = {k: v for k, v in vars(known).items() if v is not None}

subject = known.get("subject"); known.pop("subject")
df = f.load(subject)
logy = known.get("logy"); known.pop("logy")

f.setupplot()
f.regplot(df, **known)
if logy: plt.yscale('log')
f.savefig(f'{subject}regplot')

