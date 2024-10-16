#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Theo Portlock
'''
import metatoolkit.functions as f
import pandas as pd
from ast import literal_eval

# load data
meta = pd.read_excel("../data/05.LEAP_SES_CRF.xlsx", sheet_name='LEAP-SES')
labels = pd.read_csv("../data/05.LEAP_SES_CRF_MAPPING_TP.tsv", sep='\t', index_col=0)

# drop dates
meta = meta.drop(['DOB','DOI'], axis=1)

# create mappings
categories = labels.dropna()['Mapping']
mapping = categories.apply(lambda x: literal_eval('{' + x + '}'))

# map
for maps in mapping.index: meta[maps] = meta[maps].replace(mapping[maps])

# rename to new description
meta = meta.rename(columns=labels.Description)
meta = meta.set_index('Subject identification number')

# remove stratified financials - error in reporting
meta = meta.loc[:, ~meta.columns.str.startswith('House rent')]

# remove all unchanging variables or only one
meta = meta.loc[:, ~meta.columns.str.startswith('Other')]
meta = meta.loc[:, meta.nunique() != 1]

# add nutritional status
meta.loc[(meta.index.astype(str).str[0] == '1')|(meta.index.astype(str).str[0] == '3'), 'Condition'] = 'MAM'
meta.loc[meta.index.astype(str).str[0] == '2', 'Condition'] = 'Well-nourished'

# create ID
meta.index = 'LCC' + meta.index.astype(str)
meta.index = meta.index.set_names('ID')

# Format and merge Breastfeeding, Place of birth, and delivery mode data
pobirth_BF = pd.read_excel("../data/03._Bangladesh_Breast_Feeding_periods_16-Nov-2023.xlsx", sheet_name='Sheet1', index_col=0)
delivery_supp = pd.read_excel("../data/04._Bangladesh_baby_delivery_mode_and_supplements_during_pregnancy.xlsx", sheet_name='Mother Enrollment', index_col=0)
pobirth_BF.index = 'LCC' + pobirth_BF.index.astype(str) 
delivery_supp.index = 'LCC' + delivery_supp.index.astype(str)
pobirth_BF['Duration of Exclusive Breast Feeding (Months)'] = pobirth_BF['Duration of Exclusive Breast Feeding (Month.Day)']
pobirth_BF['Place of birth'] = pobirth_BF['H/O Place of birth (1=Home, 2 = Health Facility)'].map({1:'Home', 2:'Clinic'})
pobirth_BF = pobirth_BF[['Duration of Exclusive Breast Feeding (Months)','Place of birth']]
delivery_supp['Delivery Mode'] = delivery_supp['Mode'].map({1:'Vaginal',2:'Caesarean'})
delivery_supp['Supplementation during pregnancy'] = delivery_supp['Supple'].map({1:True,2:False})
delivery_supp = delivery_supp[['Delivery Mode','Supplementation during pregnancy']]
meta = pd.concat([meta, pobirth_BF, delivery_supp], join='inner', axis=1)

# Replace Yes and No with True/False
meta = meta.replace({'No':False,'Yes':True})

# Remove 3yr controls
meta = meta.loc[meta.index.str[3] != '3']

# Replace all spaces with underscores
meta.columns = meta.columns.str.replace(' ', '_')

# Remove brackets
meta.columns = meta.columns.str.replace(r'[()]', '', regex=True)

# save
meta.index = meta.index.rename('ID')
f.save(meta, 'meta')
