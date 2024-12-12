#!/home/theop/venv/bin/python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import functions as f
import os
import pandas as pd

parser = argparse.ArgumentParser(description='''
Stratify - Stratifies a dataframe according to the column of another dataframe (commonly metadata)
''')

parser.add_argument('subject')
parser.add_argument('level')
parser.add_argument('--df2')
known, unknown = parser.parse_known_args()
known = {k: v for k, v in vars(known).items() if v is not None}
unknown = eval(unknown[0]) if unknown != [] else {}

subject = known.get("subject"); known.pop('subject')
if os.path.isfile(subject): subject = Path(subject).stem
df = f.load(subject)

meta = f.load(known.get("df2")) if known.get("df2") else f.load('meta')
level = known.get('level')

output = f.stratify(df, meta, level)
print(output)
if output is not None:
    f.save(output, f'{subject}{level}')
else:
    print(subject, 'is empty')
