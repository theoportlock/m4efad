#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import functions as f
import os
import pandas as pd
import shutil
import subprocess
import sys

parser = argparse.ArgumentParser(description='''
Createsupptables - Combines tables to make final supplementary table in excel
''')

parser.add_argument('-i', '--input')
parser.add_argument('-o', '--output')
known = parser.parse_args()
known = {k: v for k, v in vars(known).items() if v is not None}

inp = known.get("intput") if known.get("intput") else 'suppTableList.txt'
out = known.get("output") if known.get("output") else 'suppTables.xlsx'

# Create Supp tables
tables = pd.read_csv(f'../figures/{inp}', header=None)
writer = pd.ExcelWriter(f'../figures/{out}')
table = '../results/' + tables + '.tsv'
with pd.ExcelWriter('../figures/suppTables.xlsx') as writer:
    for j,i in enumerate(table[0]):
        try:
            pd.read_csv(i, sep='\t', index_col=0).to_excel(
                    writer,
                    sheet_name='SuppT' + str(j+1) + '_' + i.split('/')[-1].split('.')[0])
        except:
            print('table ', i, ' not found')

