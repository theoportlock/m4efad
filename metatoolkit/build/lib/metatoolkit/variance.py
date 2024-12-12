#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import functions as f
import pandas as pd

parser = argparse.ArgumentParser(description='''
Calculates the explained variance of the categorical labels
''')
parser.add_argument('-df1', required=True)
parser.add_argument('-df2', required=False)
parser.add_argument('-p', '--pval')

known, unknown = parser.parse_known_args()
known = {k: v for k, v in vars(known).items() if v is not None}
unknown = eval(unknown[0]) if unknown != [] else {}

if known.get('df2'):
    DF1 = f.load(known.get("df1"))
    DF2 = f.load(known.get("df2"))
    output = f.explainedvariance(DF1, DF2, **known|unknown)
    print(output)
    f.save(output, f'{known.get("df1")}{known.get("df2")}power')
else:
    DF1 = f.load(known.get("df1"))
    output = f.PERMANOVA(DF1, full=True)
    print(output)
    #f.save(output, f'{known.get("df1")}power')
