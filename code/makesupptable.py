#!/usr/bin/env python
import os
import pathlib
import pandas as pd
import shutil
import subprocess
import sys

# Create Supp tables in excel
tables = pd.read_csv('../figures/suppTableList.txt', header=None)
writer = pd.ExcelWriter('../figures/suppTables.xlsx')
table = '../results/' + tables + '.tsv'
# Add Glossary
table = pd.concat([pd.Series('../figures/Supp_table_glossary.tsv'), table])
with pd.ExcelWriter('../figures/suppTables.xlsx') as writer:
    for j,i in enumerate(table[0]):
        pd.read_csv(i, sep='\t', index_col=0).to_excel(
                writer,
                sheet_name='SuppT' + str(j) + '_' + i.split('/')[-1].split('.')[0])

# Create Supp tables for latex
tables = pd.read_csv('../figures/suppTableList.txt', header=None)
table = '../results/' + tables + '.tsv'
mydir='../figures/texsupptables'
try: shutil.rmtree(mydir)
except: None
pathlib.Path(mydir).mkdir(parents=True) 
for j,i in enumerate(table[0]):
    pd.read_csv(i, sep='\t', index_col=0, dtype=object).to_latex('../figures/texsupptables/TableS' + str(j+1) + '_' + i.split('/')[-1].split('.')[0] + '.tex', escape="Latex", longtable=True)
