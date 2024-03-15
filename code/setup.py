#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Theo Portlock
For project setup
'''
import functions as f
import numpy as np
import pandas as pd

# for metadata ------------
meta1 = pd.read_excel("../data/oct_23_metadata_wellcomeleap.xlsx", sheet_name='batch1_batch2', index_col=1)
meta2 = pd.read_excel("../data/oct_23_metadata_wellcomeleap.xlsx", sheet_name='batch3', index_col=1)
meta3 = pd.read_excel("../data/oct_23_metadata_wellcomeleap.xlsx", sheet_name='batch4', index_col=1)
meta4 = pd.read_excel("../data/oct_23_metadata_wellcomeleap.xlsx", sheet_name='batch5', index_col=1)
meta = pd.concat([meta1, meta2, meta3, meta4])
controls = meta.loc[meta.index.str.contains('positive_control')]
meta = meta.loc[meta.Age_months == 12] # filter baseline only and no controls
#meta = meta.drop(['Subject_ID','Collection_date','Sequence_date', 'Age_months'], axis=1)
meta = meta.drop(['Subject_ID','Sequence_date', 'Age_months'], axis=1)

# for anthropometrics ------------
anthro1 = pd.read_excel('../data/06._LEAP_Child_Anthropometry_for_1_Yrs_MAM (Only Baseline)_06-Mar-2023.xlsx', index_col=0)
anthro2 = pd.read_excel('../data/07._LEAP-Child_Anthropometry_for_1_Yrs_Normal_16-Feb-2023.xlsx', index_col=0)
anthro2 = anthro2.loc[anthro2.An_Time == 1]
anthro = pd.concat([anthro1, anthro2])
anthro.DOM = pd.to_datetime(anthro.DOM)
anthro.DOB = pd.to_datetime(anthro.DOB)
anthro['DaysAfterBirth'] = anthro.DOM - anthro.DOB
anthro.DaysAfterBirth = anthro.DaysAfterBirth.dt.days
anthro.index = 'LCC' + anthro.index.astype(str) +'1001'
anthro['SEX'] = anthro['SEX'].astype(str).map({'1':'Male','2':'Female'})
meta = meta.join(anthro[['SEX','DaysAfterBirth']]) # update meta
anthro.drop(['SEX', 'DaysAfterBirth', 'DOV', 'DOB', 'DOM', 'Missed', 'Group','Length', 'An_Time'], axis=1, inplace=True)
f.save(anthro, 'anthro')

# for quality ------------
quality = pd.read_csv('../data/m4efad_kneaddata_read_counts_oct2023.tsv', sep='\t', index_col=0)
quality = quality.loc[:,quality.columns.str.contains('final')].sum(axis=1).div(1e6).to_frame('HQ read depth (million reads)')
quality = quality.join(meta.reset_index().set_index('Seq_ID')['Sample_Name'], how='inner').set_index('Sample_Name')
f.save(quality, 'quality')

# for species ------------
taxo = pd.read_csv('../data/m4efad_metaphlan3_profiles_oct2023.tsv', sep='\t', index_col=0, header=1)
taxo = taxo.iloc[:, 1:].T
taxo.index = taxo.index.str.replace('.metaphlan','')
taxo = taxo.loc[:,~taxo.columns.str.contains('k__Viruses')]
taxo = taxo.loc[:,~taxo.columns.str.contains('UNKNOWN')]
controltaxo = taxo.join(controls.reset_index().set_index('Seq_ID')['Sample_Name'], how='inner').set_index('Sample_Name')
taxo = taxo.join(meta.reset_index().set_index('Seq_ID')['Sample_Name'], how='inner').set_index('Sample_Name')
f.save(taxo, 'taxo')

tdf = taxo.loc[:, taxo.columns.str.contains('s__\w+$', regex=True)]
tdf.columns = tdf.columns.str.replace('.*\|s\_\_','', regex=True)
species = f.norm(tdf)
f.save(species, 'species')

tdf = taxo.loc[:, taxo.columns.str.contains('g__\w+$', regex=True)]
tdf.columns = tdf.columns.str.replace('.*\|g\_\_','', regex=True)
genus = f.norm(tdf)
f.save(genus, 'genus')

tdf = taxo.loc[:, taxo.columns.str.contains('f__\w+$', regex=True)]
tdf.columns = tdf.columns.str.replace('.*\|f\_\_','', regex=True)
family = f.norm(tdf)
f.save(family, 'family')

tdf = taxo.loc[:, taxo.columns.str.contains('o__\w+$', regex=True)]
tdf.columns = tdf.columns.str.replace('.*\|o\_\_','', regex=True)
order = f.norm(tdf)
f.save(order, 'order')

tdf = taxo.loc[:, taxo.columns.str.contains('c__\w+$', regex=True)]
tdf.columns = tdf.columns.str.replace('.*\|c\_\_','', regex=True)
clas = f.norm(tdf)
f.save(clas, 'class')

tdf = taxo.loc[:, taxo.columns.str.contains('p__\w+$', regex=True)]
tdf.columns = tdf.columns.str.replace('.*\|p\_\_','', regex=True)
phylum = f.norm(tdf)
f.save(phylum, 'phylum')

tdf = taxo.loc[:, taxo.columns.str.contains('k__\w+$', regex=True)]
tdf.columns = tdf.columns.str.replace('.*\|k\_\_','', regex=True)
kingdom = f.norm(tdf)
f.save(kingdom, 'kingdom')

# for controls
zymoD6300 = pd.Series({'Listeria_monocytogenes' : 0.139,
                       'Pseudomonas_aeruginosa_group' : 0.061,
                       'Bacillus_subtilis_group' : 0.103,
                       'Escherichia_coli' : 0.085,
                       'Salmonella_enterica' : 0.087,
                       'Lactobacillus_fermentum' : 0.216,
                       'Enterococcus_faecalis' : 0.146,
                       'Staphylococcus_aureus' : 0.152,
                       'Saccharomyces_cerevisiae' : 0.0057,
                       'Cryptococcus_neoformans' : 0.0037}, name='zymo')
controlspecies = controltaxo.loc[:,controltaxo.columns.str.contains('s__')]
controlspecies.columns = controlspecies.columns.str.replace('.*s__','', regex=True)
controlspecies = f.norm(controlspecies)
controlout = zymoD6300.to_frame().join(controlspecies.T)
f.save(controlout, 'controls')

# for function ------------
pathways = pd.read_csv("../data/m4efad_humann3_pathway_cpm_oct2023.tsv", index_col=0, sep='\t').T
pathways.index = pathways.index.str.replace('_Abundance','')
#pathways = f.norm(pathways)
#pathways = pathways.loc[:,~pathways.columns.str.contains('UNMAPPED')]
#pathways = pathways.loc[:,~pathways.columns.str.contains('UNINTEGRATED')]
#meta = f.load('meta')
pathways = pathways.join(meta.reset_index().set_index('Seq_ID')['Sample_Name'], how='inner').set_index('Sample_Name')
pathways = f.filter(pathways, nonzero=True)
f.save(pathways, 'pathways')
pathwaysall = f.norm(pathways.loc[:, ~pathways.columns.str.contains('\|')])
f.save(pathwaysall, 'pathwaysall')
pathways = pathwaysall.loc[:, ~pathwaysall.columns.str.contains('\|')]
pathways = pathways.loc[:,pathways.columns.str.contains('super')]
pathways = f.filter(pathways, prevail=0.15)
f.save(pathways, 'pathwayssuper')

speciescover = pd.read_csv('../data/m4efad_humann3_pathway_coverage_oct2023.tsv', index_col=0, sep='\t').T
speciescover.index = speciescover.index.str.replace('_Coverage','')
#speciescover = speciescover.loc[:,~speciescover.columns.str.contains('UNMAPPED')]
#speciespath = speciescover.loc[:,~speciescover.columns.str.contains('UNINTEGRATED')]
#speciescover= f.filter(speciescover, nonzero=True)
speciescover = speciescover.join(meta.reset_index().set_index('Seq_ID')['Sample_Name'], how='inner').set_index('Sample_Name')
f.save(speciescover, 'coverage')

# for metabolomics ------------
dfp = pd.read_excel("../data/LEAP_01_&_02_combined_Theo_v03.xlsx", sheet_name='POS', index_col=0).iloc[1:,:]
dfn = pd.read_excel("../data/LEAP_01_&_02_combined_Theo_v03.xlsx", sheet_name='NEG', index_col=0).iloc[1:,:]
metabid = pd.read_excel("../data/LEAP_01_&_02_combined_Theo_v03.xlsx", sheet_name='sample IDs', index_col=0)
dfn.index = dfn.index.str.replace(r'.*_S','', regex=True).str.replace(r'.*_Q','Q', regex=True).str.replace(r'_RT_shift','', regex=True)
dfp.index = dfp.index.str.replace(r'.*_S','', regex=True).str.replace(r'.*_Q','Q', regex=True).str.replace(r'_RT_shift','', regex=True)
dfn = dfn.loc[~dfn.index.str.startswith('Q')]
dfp = dfp.loc[~dfp.index.str.startswith('Q')]
dfn.index = dfn.index.astype(int)
dfp.drop('106B', inplace=True)
dfp.index = dfp.index.astype(int)
dfp = dfp.reset_index().rename(columns={'Unnamed: 1':'treatment', 'index':'Liggins sample'}).set_index(['treatment','Liggins sample'])
dfn = dfn.reset_index().rename(columns={'Unnamed: 1':'treatment', 'index':'Liggins sample'}).set_index(['treatment','Liggins sample'])
dfn.columns = dfn.columns.str.replace('\|.*','', regex=True)
dfp.columns = dfp.columns.str.replace('\|.*','', regex=True)
df = pd.concat([dfp,dfn], join='inner', axis=1)
df = df.groupby(level=0, axis=1).max()
metabid['sample'] = metabid['sample'].str[:-4] + '1001'
metabid['Liggins sample'] = metabid['Liggins sample'].str.extract('(\d+)').astype(int)
metabid.treatment = metabid.treatment.str.upper()
metabid.set_index(['treatment','Liggins sample'], inplace=True)
metab = df.join(metabid['sample'], how='inner').set_index('sample')
df = metab.astype(float)
# filtering names
df = df.loc[:,df.columns.str.contains('\:')]
df = df.loc[:,~df.columns.str.contains('nsettled')]
df = df.loc[:,~df.columns.str.contains('Unnamed')]
# filter other timepoints
metab = df.join(meta['Condition'], how='inner').drop('Condition', axis=1)
metab = metab.round(6)
m = metab.columns[0]
# PREVAILANCE
for m in metab.columns:
    zerovals = metab[m].value_counts().sort_values().gt(1)
    zerovals = zerovals.loc[zerovals].index.to_list()
    for v in zerovals:
        metab.loc[metab[m] == v, m] = 0
prevail = 0.1 # filter for those that are present in over 80% of samples
st = f.stratify(metab, meta, 'Condition')
ch = f.change(st)['MAMvsHealthy']
prevs = ch.loc[:, ch.columns.str.contains('prev')]
metab = metab.drop(prevs.loc[prevs.lt(prevail).any(axis=1)].index, axis=1)

# Add pseudovalue for 0s
for m in metab.columns:
    minval = metab.loc[metab[m] != 0, m].min()
    metab.loc[metab[m] == 0, m] = minval/5
metab = metab.apply(np.log1p) # APPLY LOG - keep above 0
f.save(metab, 'metab')

# for lipid mapping
metab = f.load('metab')
lipids = pd.read_csv('../data/lipids.tsv', sep='\t', index_col=0, dtype=object)
abbrev = lipids.dropna(subset='ABBREVIATION').set_index('ABBREVIATION')
abbrev = abbrev.groupby(level=0).first()
metab = metab.T.join(abbrev.dropna(subset='MAIN_CLASS')['MAIN_CLASS'].groupby(level=0).first(), how='inner').set_index('MAIN_CLASS').T
metab = metab.groupby(level=0, axis=1).sum()
f.save(metab, 'lipids')

# for EEG ------------
eeg = pd.read_csv("../data/DhakaBangladeshLEAPE-EEGProcessedVariable_DATA_LABELS_2023-07-10_1606.csv", index_col=0).iloc[:,2:]
eeg.index += '1001'
eeg.columns = eeg.columns.str.replace(' Alpha','-Alpha')
eeg = eeg.join(meta['Seq_ID'], how='inner').drop('Seq_ID', axis=1)
eeg = eeg.dropna()
eeg.columns = eeg.columns.str.replace(' $','', regex=True)
f.save(eeg, 'eeg')

# for bayley ------------
bayley = pd.read_csv("../data/DhakaBangladeshLEAPE-BayleyCompletes_DATA_LABELS_2023-02-23_1725.csv", index_col=0)
bayley.index += '1001'
bayley = bayley.loc[:,bayley.columns.str.contains('Raw')]
bayley.columns = bayley.columns.str.replace(' Raw','')
bayley = bayley.join(meta['Seq_ID'], how='inner').drop('Seq_ID', axis=1)
f.save(bayley, 'bayleys')

# for wolke ------------
wolkes = pd.read_csv('../data/DhakaBangladeshLEAPE-Wolkes_DATA_LABELS_2023-05-10_1935.csv', index_col=0, header=1)
wolkes = wolkes.loc[wolkes['Unnamed: 2'] == '12 months']
wolkes = wolkes.loc[:, wolkes.columns.str.contains('wolke')]
wolkes.columns = wolkes.columns.str.replace('_bangla','')
wolkes.columns = wolkes.columns.str.replace('_bangle','')
wolkes.index += '1001'
wolkes = wolkes.join(meta['Seq_ID'], how='inner').drop('Seq_ID', axis=1)
f.save(wolkes, 'wolkes')

# for othermeta ------------
othermeta = pd.read_excel("../data/05.LEAP_SES_CRF.xlsx", index_col=0)
othermeta.index = "LCC" + othermeta.index.astype(str) + "1001"
othermeta['Ethnicit'] = othermeta.Ethnicit.replace({1:'Bangali', 2:'Bihari', 3:'Others'})
othermeta['Fam_Type'] = othermeta.Fam_Type.replace({1:'Nuclear', 2:'Joint'})
othermeta['Language'] = othermeta.Lan.replace({1:'Bangali', 2:'Urdu', 3:'Others'})
othermeta['Maid'] = othermeta.Maid.sub(2).abs()
occmap = { 1:"Dependent (< 5 years)", 2:"Student", 3:"Housewife", 4:"Government Service", 5:"Private Service / NGO Service", 6:"Small Businessman (Up to 10,000 Tk./ Month)", 7:"Medium Businessman (10,001 – 30,000 Tk./ Month)", 8:"Businessman (> 30,000 Tk./ Month)", 9:"Landlord / House owner", 10:"Contractor (Construction work)", 11:"Home skill worker (Shari made, Embroidery, Karchupi etc.)", 12:"Skilled labourer (Radio, TV, Fan, Plumber, Mason, Painter, Electrician/Lineman, Gas, Phone, WASA, Industry worker etc.)", 13:"Daily labourer (unskilled labourer)", 14:"Rickshaw /Push cart puller/van", 15:"Garments worker", 16:"Farmer", 17:"Hawker", 18:"Driver", 19:"Helper", 20:"Sweeper", 21:"Barber", 22:"Cobbler", 23:"Beggar", 24:"Tailor", 25:"Cook", 26:"Teacher (School, College, University)", 27:"Clergy (Imam, Muarzin, Moulana, Huzur, Padri etc)", 28:"Doctor (MBBS, MD, FCPS, FRCP etc.)", 29:"Engineer (B. Sc., M.Sc., PhD etc.)", 30:"Lawyer", 31:"Servant", 32:"Retired person", 33:"Unemployed", 77:"Other (Specify)"}
othermeta['Moth_Occ'] = othermeta.Moth_Occ.replace(occmap)
othermeta['Fath_Occ'] = othermeta.Fath_Occ.replace(occmap)
othermeta['Own_Hous'] = othermeta['Own_Hous'].sub(2).abs()
othermeta['H_Rent'] = othermeta.H_Rent.replace({0:np.nan})
othermeta['Floor'] = othermeta.Floor.replace({1:"Earth/Sand/Clay/Mud/Dung", 2:"Wood", 3:"Ceramic Tiles", 4:"Cement/Concrete", 5:"Other"})
othermeta['Wall'] = othermeta.Wall.replace({1:'Bamboo/cane/straw', 2:'Rudimentary walls (mud) or mixed with mud', 3:'Tin wall', 4:'Bricks, cement blocks, concrete walls', 5:'wood'})
othermeta['Roof'] = othermeta.Roof.replace({1:'Natural material (straw)', 2:'Rudimentary roofing', 3:'Finished roof (concrete)', 4:'Tin roof'})
othermeta['Electrct'] = othermeta['Electrct'].sub(2).abs()
othermeta['Gas'] = othermeta['Gas'].sub(2).abs()
othermeta['Mobile'] = othermeta['Mobile'].sub(2).abs()
othermeta['Almeria'] = othermeta['Almeria'].sub(2).abs()
othermeta['Table'] = othermeta['Table'].sub(2).abs()
othermeta['Chair'] = othermeta['Chair'].sub(2).abs()
othermeta['Bench'] = othermeta['Bench'].sub(2).abs()
othermeta['Clock'] = othermeta['Clock'].sub(2).abs()
othermeta['Bed'] = othermeta['Bed'].sub(2).abs()
othermeta['Radio'] = othermeta.Radio.sub(2).abs()
othermeta['TV'] = othermeta.TV.sub(2).abs()
othermeta['Bicycle'] = othermeta['Bicycle'].sub(2).abs()
othermeta['Bike'] = othermeta['Bike'].sub(2).abs()
othermeta['SMachine'] = othermeta['SMachine'].sub(2).abs()
othermeta['Fan'] = othermeta['Fan'].sub(2).abs()
othermeta['ReadNews'] = othermeta['ReadNews'].replace({1:0, 2:1, 3:2})
othermeta['Radio_TV'] = othermeta['Radio_TV'].replace({1:0, 2:1, 3:2})
othermeta['Social_M'] = othermeta['Social_M'].replace({1:0, 2:1, 3:2})
othermeta['Dr_Water'] = othermeta['Dr_Water'].replace({1:"Municipality supply/piped water", 2:"Own arrangement by pump", 3:"Tube well", 4:"Well/Pond/Canal)"})
othermeta['Toilet'] = othermeta['Toilet'].replace({1:'Septic tank or toilet', 2:'Water-sealed or slab latrine', 3:'Pit latrine', 4:'Open latrine', 5:'Hanging latrine', 6: 'Bush, field as latrine'})
othermeta['Food'] = othermeta['Food'].replace({1:"Deficit in whole year", 2:"Sometimes deficit", 3:"Neither deficit nor surplus", 4:"Surplus"})
othermeta['BFIAGENT'] = othermeta['BFIAGENT'].replace({1:"Water", 2:"Mud", 3:"Ash", 4:"Soap", 5:"Other", 9:"None"})
othermeta['BFIHOW'] = othermeta['BFIHOW'].replace({1:"Left hand", 2:"Right hand", 3:"Both hands", 9:"None"})
othermeta['BFISoW'] = othermeta['BFISoW'].replace({1:"Municipality supply/piped water", 2:"Own arrangement by pump", 3:"Tube well", 4:"Well/Pond/Canal", 9:"None"})
othermeta['BEAGENT'] = othermeta['BEAGENT'].replace({1:"Water", 2:"Mud", 3:"Ash", 4:"Soap", 5:"Other", 9:"None"})
othermeta['BEHOW'] = othermeta['BEHOW'].replace({1:"Left hand", 2:"Right hand", 3:"Both hands", 9:"None"})
othermeta['BE_SoW'] = othermeta['BE_SoW'].replace({1:"Municipality supply/piped water", 2:"Own arrangement by pump", 3:"Tube well", 4:"Well/Pond/Canal", 9:"None"})
othermeta['ADAGENT'] = othermeta['ADAGENT'].replace({1:"Water", 2:"Mud", 3:"Ash", 4:"Soap", 5:"Other", 9:"None"})
othermeta['ADHOW'] = othermeta['ADHOW'].replace({1:"Left hand", 2:"Right hand", 3:"Both hands", 9:"None"})
othermeta['ADSoW'] = othermeta['ADSoW'].replace({1:"Municipality supply/piped water", 2:"Own arrangement by pump", 3:"Tube well", 4:"Well/Pond/Canal", 9:"None"})
othermeta['BCDAGENT'] = othermeta['BCDAGENT'].replace({1:"Water", 2:"Mud", 3:"Ash", 4:"Soap", 5:"Other", 9:"None"})
othermeta['BCDHOW'] = othermeta['BCDHOW'].replace({1:"Left hand", 2:"Right hand", 3:"Both hands", 9:"None"})
othermeta['BCDSoW'] = othermeta['BCDSoW'].replace({1:"Municipality supply/piped water", 2:"Own arrangement by pump", 3:"Tube well", 4:"Well/Pond/Canal", 9:"None"})
othermeta['ACAAGENT'] = othermeta['ACAAGENT'].replace({1:"Water", 2:"Mud", 3:"Ash", 4:"Soap", 5:"Other", 9:"None"})
othermeta['ACAHOW'] = othermeta['ACAHOW'].replace({1:"Left hand", 2:"Right hand", 3:"Both hands", 9:"None"})
othermeta['ACASoW'] = othermeta['BCDSoW'].replace({1:"Municipality supply/piped water", 2:"Own arrangement by pump", 3:"Tube well", 4:"Well/Pond/Canal", 9:"None"})
othermeta['Water_TM'] = othermeta['Water_TM'].replace({1:"None", 2:"Let it stand & settle", 3:"Water filter", 4:"Solar disinfection", 5:"Boil", 6:"Strain through cloth", 7:"Add bleach/Chlorine"})
othermeta['Toilet_F'] = othermeta['Toilet_F'].sub(2).abs()
othermeta['Cook_Plc'] = othermeta['Cook_Plc'].replace({1:"Inside house", 2:"Separate building", 3:"Outdoors", 4:"Other"})
othermeta['C_Fuel'] = othermeta['C_Fuel'].replace({1:"Gas", 2:"Electric stove", 3:"Wood", 4:"Kerosene oil stove", 5:"Animal dung", 6:"Garments products"})
othermeta['Drain'] = othermeta['Drain'].sub(2).abs()
othermeta = othermeta.drop(['DOB','Other_Et', 'OtherLan', 'DOI', 'OFO', 'OMO','Lan', 'Sex'], axis=1)
meta = meta.join(othermeta, how='inner')
#meta = meta.drop(['Seq_ID'], axis=1)
meta['Condition'] = meta['Condition'].replace('Healthy','Well-nourished')
meta['Condition'] = meta['Condition'].replace('MAM','Malnourished')
meta.Seq_batch = meta.Seq_batch.astype(int)
f.save(meta,'meta')

