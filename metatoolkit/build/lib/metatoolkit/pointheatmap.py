#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import functions as f

parser = argparse.ArgumentParser(description='''
Pointheatmap - Produces a pointheatmap of a given dataset
''')
parser.add_argument('subject')
known = parser.parse_args()
known = {k: v for k, v in vars(known).items() if v is not None}

subject = known.get("subject")
df = f.load(subject)

f.setupplot()
f.pointheatmap(df)
f.savefig(f'{subject}heatmap')
