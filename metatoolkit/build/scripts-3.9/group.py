#!/home/theop/venv/bin/python3
# -*- coding: utf-8 -*-

import argparse
import functions as f
import pandas as pd

parser = argparse.ArgumentParser(description='Group - Groups a dataset')
parser.add_argument('subject')
parser.add_argument('--on_index', action='store_true')
parser.add_argument('--index_levels', type=int, default=1)
parser.add_argument('--axis')
parser.add_argument('--func')
parser.add_argument('-o', '--output')

known = parser.parse_args()
known = {k: v for k, v in vars(known).items() if v is not None}

def group(df, on_index=True, axis=None, func=None):
    if on_index:
        outdf = df.groupby(level=0).agg(func=func, axis=axis)
    else:
        outdf = df.agg(func=func, axis=axis).to_frame(f'{subject}{func}')
    return outdf

subject = known.get("subject")
output = known.get("output")
on_index = known.get("on_index")
func = known.get("func")
axis = known.get("axis")
index_levels = known.get("index_levels")

df = f.load(subject)

if index_levels > 1:
    df = df.reset_index()
    df = df.set_index(df.columns[:index_levels].to_list())

out = group(df, on_index=on_index, axis=axis, func=func)
print(out)

if known.get("output"):
    f.save(out, output)
else:
    f.save(out, subject+func)
