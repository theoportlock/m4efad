#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import functions as f
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser(description='''
Abund - Produces an Abundance/Relative abundance plot of a given dataset
''')
parser.add_argument('subject')
known = parser.parse_args()
known = {k: v for k, v in vars(known).items() if v is not None}

subject = known.get("subject")
df = f.load(subject)

f.setupplot()
f.abund(df)
f.savefig('{subject}abund')
