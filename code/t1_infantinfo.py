#!/usr/bin/env python
import pandas as pd
import numpy as np
import functions as f
import matplotlib.pyplot as plt

meta = f.load('meta')
df = f.load('anthro')
df['Sex: Female'] = df.join(meta['SEX'], how='inner').SEX.eq('Female').astype(int)
dums = pd.get_dummies(meta.drop(['H_Rent', 'Condition'], axis=1))
df = df.join(dums, how='inner')
condition = f.stratify(df, meta, 'Condition')
boolcols = condition.columns[condition.nunique() == 2]
notboolcols = condition.columns[condition.nunique() > 3]

# Calculates significant non-categorical covariates
change = f.change(condition[notboolcols], analysis=['prevail', 'mww'])['MalnourishedvsWell-nourished']
change['Well-nourished'] = change['Well-nourishedmean'].round(2).astype(str) + ' ± ' + change['Well-nourishedstd'].round(2).astype(str)
change['Malnourished'] = change.Malnourishedmean.round(2).astype(str) + ' ± ' + change.Malnourishedstd.round(2).astype(str)
change['pval'] = change['MWW_pval']
mappings = {'Weight':'Weight (kg)',
            'MUAC':'MUAC (cm)',
            'HC':'HC (cm)',
            'WLZ_WHZ':'WLZ_WHZ (z-score)',
            'DaysAfterBirth':'Days after birth',
            'Oth_Exp':'Other expenses (eg. medical, educational)(taka)',
            'Live_Cur':'How long lived in current household (years)',
            'Inc_Moth':'Mothers income (taka)',
            'Years_of_Father_Education':'Years of Father Education',
            'Fath_Edu':'Fathers education level',
            'Expend':'Monthly Total expenditure (taka)',
            'Inc_HH':'Household head’s income (taka)',
            'Fam_Exp':'Family expenditure (taka)',
            'B_Order':'Birth order of enrolled child among live births',
            'Mem_Mob':'Household members using mobile phone',
            'T_Income':'Total monthly Income (taka)',
            'Sibl_5Y':'Number of siblings under 5 years',
            'Moth_Edu':'Mother education level',
            'Years_of_Mother_Education':'Years of Mother Education',
            'Member':'Number of members in your household',
            'Sleep_Hh':'Number of people sleeping in household',
            'Child':'Number of living children',
            'Cur_Room':'Number of rooms in current household',
            'Inc_OM':'Other member’s income (taka)',
            'Inc_OS':'Other sources income (taka)'}
change.index = change.index.to_series().replace(mappings)
patientinfo = change[['Well-nourished','Malnourished','pval']]

fboolcols = []
for con in condition[boolcols].columns:
    if condition[con].value_counts().min() > 50:
        fboolcols.append(con)

counting = condition[fboolcols].groupby(level=0).sum().astype(int).astype(str) + ' (' + condition[fboolcols].groupby(level=0).sum().div(condition[fboolcols].groupby(level=0).count()).mul(100).round(1).astype(str) + '%)'
counting = counting.drop(['Almeria','Table','Chair','Clock','TV','SEX_Female', 'SEX_Male','Fam_Type_Joint','Wall_Bricks, cement blocks, concrete walls','Wall_Tin wall','Roof_Finished roof (concrete)','Roof_Tin roof','Food_Sometimes deficit','Food_Surplus','BFIAGENT_Water','BEAGENT_Water','Water_TM_None','Cook_Plc_Inside house'], axis=1).T

# Calculates significant categorical covariates
fish = f.fisher(condition[boolcols].reset_index())
sig = fish.loc['Condition'].sort_values('pvals').dropna()
sig['logodds'] = sig.odds.apply(np.log).dropna()
fsig = sig.loc[sig.pvals.lt(0.05)]
fsig.dropna(inplace=True)
#fsig.logodds.sort_values().plot.barh()
#plt.show()

out = counting.join(sig['pvals'], how='inner')
out = out.sort_values('pvals')
out['pval']=out.pvals.round(2)
out = out.drop(['pvals'], axis=1)
out.index = out.index.to_series().replace({'BEAGENT_Soap':'Before eating: Soap','BFIAGENT_Soap':'Before feeding child: Soap','Water_TM_Boil':'Water treatment method: Boil','Toilet_Septic tank or toilet':'Principal toilet used: Septic-tank/toilet', 'Cook_Plc_Outdoors':'Place of Cooking: Outdoors',' Fath_Occ_Daily labourer (unskilled labourer)':'Fathers occupation','Wall_Tin wall':'Household: Tin walls', 'Drain':'Household: Drain', 'Own_Hous':'Household: Own house','Condition_MAM':'Malnutrition - MAM','Cook_Plc_Inside house':'Cooking inside/outside home','Water_TM_None':'Water Treatment: None','SEX_Male':'Sex','Clock':'Household: clock','Fam_Type_Nuclear':'Nuclear family','Wall_Bricks, cement blocks, concrete walls':'Household: Brick walls','Toilet_F':'Toilet facility shared with other households','Moth_Occ_Housewife':'Mothers occupation: Housewife','BEHOW_Both hands':'Sanitation before feeding - mother: both hands','BFIHOW_Both hands':'Sanitation before feeding: both hands','BFISoW_Municipality supply/piped water':'Sanitation before feeding: piped water'})

final = pd.concat([out, patientinfo]).sort_values('pval')
final = final.set_axis((condition.index.unique() + ' (n=' + condition.index.value_counts().astype(str) + ')').to_list() + ['pval'], axis=1)
final['pval'] = final.pval.round(3)
#final['pval'] = final.pval.apply('{:0<3}'.format)
final['pval'] = final.pval.apply('{:0<5}'.format)
final.loc[final['pval'] == '0.000', 'pval'] = '<0.001'
f.save(final, 'patientinfo')
