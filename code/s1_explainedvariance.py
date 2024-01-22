#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Theo Portlock
'''
import functions as f
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load data  ------------
meta = f.load('meta')
species = f.load('species')
pathways = f.load('pathwaysall')
bayleys = f.load('bayleys')
wolkes = f.load('wolkes')
metab = f.load('metab')
eeg = f.load('eeg')

# Filter only categories
cats = meta.select_dtypes('object')
cats = pd.get_dummies(cats)
nums = meta.select_dtypes(['float64', 'int64'])
nums = nums[nums.columns[[nums[col].isin([0,1]).all() for col in nums]]]
fmeta = pd.concat([cats,nums], axis=1).astype(bool)
datasets = {'species':species, 'pathways':pathways, "bayleys":bayleys, 'wolkes': wolkes, 'metab':metab, 'eeg':eeg}
joined = f.merge(datasets)

# filter those categories that are present in at least 20% of total joined datasets
drop=[]
for col in fmeta.columns:
    minlt20 = fmeta.loc[joined.index, col].value_counts().min() < (fmeta.loc[joined.index, col].shape[0] * 0.15)
    less2cat = fmeta.loc[joined.index, col].nunique() < 2
    if minlt20 or less2cat:
        drop.append(col)
meta = fmeta.drop(drop, axis=1)

# Calculate power
targets = meta.columns
target = targets[3]
output = pd.DataFrame(index=datasets.keys())
pval=True
f.setupplot(figsize=(10,120), agg=False)
for dataset in datasets:
    for target in targets:
        tdf = datasets[dataset].join(meta[target].dropna(), how='inner').set_index(target)
        output.loc[dataset,target] = f.PERMANOVA(tdf, pval=pval)
power = -output.apply(np.log)
f.clustermap(power.T, power.T.gt(-np.log(0.05)), square=True)

plotdf = power.drop(['Almeria','Chair','TV','Toilet_Septic tank or toilet','Roof_Finished roof (concrete)','Roof_Tin roof','Table','SEX_Female','Condition_Well-nourished','Food_Surplus','Food_Sometimes deficit','BFIAGENT_Water','BFIAGENT_Soap','BEAGENT_Water','BEAGENT_Soap','Cook_Plc_Outdoors','BCDAGENT_Water','BCDAGENT_Soap','Fam_Type_Joint','Water_TM_Boil'], axis=1)

plotdf = plotdf.rename({'Fath_Occ_Daily labourer (unskilled labourer)':'Fathers occupation','Drain':'Household: Drain','Wall_Tin wall':'Household: Tin walls', 'Own_Hous':'Household: Own house','Condition_MAM':'Malnutrition - MAM','Cook_Plc_Inside house':'Cooking inside/outside home','Water_TM_None':'Water Treatment: None','SEX_Male':'Sex','Clock':'Household: clock','Fam_Type_Nuclear':'Nuclear family','Wall_Bricks, cement blocks, concrete walls':'Household: Brick walls','Toilet_F':'Toilet facility shared with other households','Moth_Occ_Housewife':'Mothers occupation','BEHOW_Both hands':'Sanitation before feeding - mother: both hands','BFIHOW_Both hands':'Sanitation before feeding: both hands','BFISoW_Municipality supply/piped water':'Sanitation before feeding: piped water', 'Sex':'Sex: Female'}, axis=1)

plotdf = plotdf.rename({'species':'Gut microbiome species','pathways':'Gut microbiome function','eeg': 'EEG','metab':'Plasma lipids', 'wolkes':'Wolkes'}, axis=0)

f.clustermap(plotdf.T, plotdf.abs().gt(-np.log(0.05)).T, figsize=(3,5.5))
f.savefig('power')
