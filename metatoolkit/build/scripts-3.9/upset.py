#!/home/theop/venv/bin/python3
# -*- coding: utf-8 -*-

import argparse
import functions as f
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path

parser = argparse.ArgumentParser(description='''
Upset - Produces an upset plot of indecies of multiple datasets
''')
parser.add_argument('datasets', nargs='+')
known = parser.parse_args()
known = {k: v for k, v in vars(known).items() if v is not None}

dfs = known.get("datasets")
alldfs = {df:f.load(df).index for df in dfs}

#f.setupplot()
f.upset(alldfs)
f.savefig(f'upset')

