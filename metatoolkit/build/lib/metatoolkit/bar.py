#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import functions as f
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser(description='''
Bar - Produces a barplot of a given dataset
''')

parser.add_argument('subject')
parser.add_argument('-x')
parser.add_argument('-y')
parser.add_argument('-hue')

known = parser.parse_args()
known = {k: v for k, v in vars(known).items() if v is not None}

subject = known.get("subject")
df = f.load(subject)

f.setupplot()
f.bar(df, **known)
f.savefig('{subject}bar')
