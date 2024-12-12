#!/home/theop/venv/bin/python3
# -*- coding: utf-8 -*-

import argparse
import functions as f
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path

parser = argparse.ArgumentParser(description='''
Heatmap - Produces a heatmap of a given dataset
''')
parser.add_argument('subject')
known = parser.parse_args()
known = {k: v for k, v in vars(known).items() if v is not None}

subject = known.get("subject"); known.pop('subject')
if os.path.isfile(subject): subject = Path(subject).stem
df = f.load(subject)

f.setupplot()
f.spindle(df)
f.savefig(f'{subject}Spindle')

