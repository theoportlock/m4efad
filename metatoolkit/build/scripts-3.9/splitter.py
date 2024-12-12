#!/home/theop/venv/bin/python3
# -*- coding: utf-8 -*-

import argparse
import functions as f

parser = argparse.ArgumentParser(description='''
Splitter - splits dataframes according to the values in a defined column''')
parser.add_argument('subject')
parser.add_argument('column')
parser.add_argument('--df2', required=False)

known, unknown = parser.parse_known_args()
known = {k: v for k, v in vars(known).items() if v is not None}
unknown = eval(unknown[0]) if unknown != [] else {}

df = f.load(known.get('subject'))
df2 = f.load(known.get('df2')) if known.get("df2") else None
col = known.get('column')
subject = known.get('subject')

output = f.splitter(df, df2, col)
print(output)

for level in output:
    f.save(output[level], f'{subject}{col}{level}')
