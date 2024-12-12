#!/home/theop/venv/bin/python3
# -*- coding: utf-8 -*-

import argparse
import functions as f

parser = argparse.ArgumentParser(description='''
Corr - Produces a report of the significant correlations between data
''')
parser.add_argument('subject', nargs='+')
parser.add_argument('-m', '--mult', action='store_true')

known = parser.parse_args()
known = {k: v for k, v in vars(known).items() if v is not None}

subject = known.get("subject")
mult = known.get("mult") if known.get('mult') else False

if len(subject) == 1:
    df = f.load(subject[0])
    output = f.corr(df)
    print(output)
    f.save(output, subject[0]+'corr')
elif len(subject) == 2:
    df1 = f.load(subject[0])
    df2 = f.load(subject[1])
    output = f.corrpair(df1, df2)[0]
    print(output)
    f.save(output, subject[0] + subject[1] +'corr')
else:
    print('invalid number of arguments')

