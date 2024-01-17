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

# For Species ------------
meta = f.load('meta')
taxo = f.load('taxo')
coverage = f.load('coverage')
pathways = f.load('pathways')
bayleys = f.load('bayleys')
metab = f.load('metab')
eeg = f.load('eeg')

cats = meta.select_dtypes('object')
cats = pd.get_dummies(cats)

nums = meta.select_dtypes(['float64', 'int64'])
nums = nums[nums.columns[[nums[col].isin([0,1]).all() for col in nums]]]
fmeta = pd.concat([cats,nums], axis=1).astype(bool)
totals = fmeta.sum().sort_values()
ftotals = totals.loc[(totals.gt(40) & (totals.lt(200)))]
meta = fmeta.loc[:,ftotals.index]

datasets = {'taxo':taxo, 'coverage':coverage, 'pathways':pathways, "bayleys":bayleys, 'metab':metab, 'eeg':eeg}
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

plotdf = power.drop(['Almeria','Chair','TV','Gas','C_Fuel_Gas','Toilet_Septic tank or toilet','Roof_Finished roof (concrete)','Roof_Tin roof','Table','ADAGENT_Soap','SEX_Female','Condition_Healthy','Food_Surplus','Food_Sometimes deficit','BFIAGENT_Water','BFIAGENT_Soap','BEAGENT_Water','BEAGENT_Soap','Cook_Plc_Outdoors','BCDAGENT_Water','BCDAGENT_Soap','Fam_Type_Joint','C_Fuel_Gas','Gas','Water_TM_Boil'], axis=1)

plotdf = plotdf.drop(['coverage'], axis=0)

plotdf = plotdf.rename({'Fath_Occ_Daily labourer (unskilled labourer)':'Fathers occupation','Wall_Tin wall':'Household: Tin walls', 'Own_Hous':'Household: Own house','Condition_MAM':'Malnutrition - MAM','Cook_Plc_Inside house':'Cooking inside/outside home','Water_TM_None':'Water Treatment: None','SEX_Male':'Sex','Clock':'Household: clock','Fam_Type_Nuclear':'Nuclear family','Wall_Bricks, cement blocks, concrete walls':'Household: Brick walls','Toilet_F':'Toilet facility shared with other households','Moth_Occ_Housewife':'Mothers occupation','BEHOW_Both hands':'Sanitation before feeding - mother: both hands','BFIHOW_Both hands':'Sanitation before feeding: both hands','BFISoW_Municipality supply/piped water':'Sanitation before feeding: piped water'}, axis=1)

plotdf = plotdf.rename({'taxo':'Gut microbiome species','pathways':'Gut microbiome function','eeg': 'EEG','metab':'Plasma lipids'}, axis=0)

f.clustermap(plotdf.T, plotdf.abs().gt(3).T)
plt.tight_layout()
f.savefig('power')
