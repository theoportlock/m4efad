#!/usr/bin/env python
import pandas as pd
import numpy as np
import metatoolkit.functions as f
import matplotlib.pyplot as plt

# Load
meta = f.load('meta')
lipids = f.load('lipids')
lipid_classes = f.load('lipid_classes')
lipid_classes.columns = lipid_classes.columns.str.replace(r'[\[\]\;\/\|\(\)\:\-\ ]','.', regex=True)
lipids.columns = lipids.columns.str.replace(r'[\[\]\;\/\|\(\)\:\-\ ]','.', regex=True)

# grouped by lipid class
lipid_classcondition = f.stratify(lipid_classes, meta, 'Condition')
lipid_classconditionchange = f.load('lipid_classeschange')
f.setupplot(figsize=(8,2))
sig = f.filter(lipid_classconditionchange, query='qval < 0.25')
filt = lipid_classcondition.loc[:,sig.index]
filt.columns=filt.columns.str.replace(r'\.\..*','', regex=True).str.replace(r'and.*','',regex=True).str.replace(r'\.', ' ', regex=True)
f.multibox(filt, sharey=False)
f.savefig('lipidclasschangebox')

# Lysolipid changes
lipids = f.load('lipids')
lysolipids = f.filter(lipids, colfilt='^L.*').sum(axis=1).to_frame('lysolipids')
lysolipidscondition = f.stratify(lysolipids, meta, 'Condition')
cats = f.load('categories')
lch = f.change(lysolipids, df2=cats)
f.setupplot(figsize=(1.5,1.5))
f.box(lysolipidscondition)
f.savefig('lysolipidsbox')

# OCFA changes
metab = f.load('lipids')
fa = metab.loc[:,metab.columns.str.startswith('FA ')]
fa.columns = fa.columns.str.replace('FA ', '')
fa.columns = fa.columns.str.replace(':.*', '', regex=True)
fa.columns = fa.columns.astype(int)
ecfa = fa.loc[: , fa.columns[fa.columns % 2 == 0]]
ecfa = ecfa.sum(axis=1).to_frame('ECFA')
ecfacondition = f.stratify(ecfa, meta, 'Condition')
ocfa = fa.loc[: , fa.columns[fa.columns % 2 != 0]]
ocfa = ocfa.sum(axis=1).to_frame('OCFA')
f.setupplot(figsize=(1.5,1.5))
ocfacondition = f.stratify(ocfa, meta, 'Condition')
f.box(ocfacondition)
f.savefig('ocfabox')

# Save OCFA and Lysolipids
extralipids = ocfa.join(lysolipids)
f.save(extralipids, 'extralipids')
