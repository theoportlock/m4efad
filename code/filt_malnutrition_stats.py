#!/usr/bin/env python
import pandas as pd
import metatoolkit.functions as f

# Load data
df = f.load('covariateedges').reset_index()

# Filter data
table = (f.filter(df,
                 query='source == "Condition.MAM"',)
                .dropna()
                .drop('source', axis=1)
                .loc[~df.target.str.contains('No')]
                .loc[~df.target.str.contains('nan')]
                .loc[~df.target.str.contains('Household')]
                .loc[~df.target.str.contains('Well-nourished')]
                .loc[~df.target.str.contains('attainment')]
                .set_index('target')
                .drop('qval', axis=1)
                .sort_values('pval'))

# Format data
#table['qval'] = table.qval.round(3)
#table['pval'] = table.pval.round(3)
#table['qval'] = table.qval.apply('{:0<5}'.format)
#table['pval'] = table.pval.apply('{:0<5}'.format)
table['effect'] = table.effect.round(3)

# Sort order of columns and rename
#table = table[['source_true_summary','source_false_summary','qval']]
table = table[['source_true_summary','source_false_summary','pval']]
table = table.rename(columns={'source_true_summary':'Malnourished','source_false_summary':'Well-nourished'})

# Save full
fulltable = table.copy()
#fulltable.loc[fulltable['qval'] == '0.000', 'qval'] = '<0.001'
#fulltable.loc[fulltable['pval'] == '0.000', 'pval'] = '<0.001'
f.save(fulltable, 'patientinfo')

# Save short
shorttable = table.copy()
#shorttable.qval = shorttable.qval.astype(float)
shorttable.pval = shorttable.pval.astype(float)
#shorttable = shorttable.query('qval <= 0.05')
shorttable = shorttable.query('pval <= 0.05')
shorttable = pd.concat([
    shorttable,
    fulltable.loc[['Sex_of_the_Child.Male',
                   'Duration_of_Exclusive_Breast_Feeding_Months',
                   'Delivery_Mode.Caesarean']]
    ])
#shorttable['qval'] = shorttable.qval.apply('{:0<5}'.format)
#shorttable['pval'] = shorttable.pval.apply('{:0<5}'.format)
#shorttable.loc[shorttable['pval'] == '0.000', 'pval'] = '<0.001'
shorttable.index = shorttable.index.str.replace('_', ' ', regex=False)
shorttable.index = shorttable.index.str.replace('.', ':', regex=False)
shorttable.index = shorttable.index.str.replace('\(.*\)', '', regex=True)
shorttable = shorttable.rename(index={'WLZ WHZ': 'WLZ/WHZ',
                                      'MUAC':'MUAC (cm)',
                                      'Duration of Exclusive Breast Feeding Months':'Months of exclusive breast feeding',
                                      'Weight':'Weight (Kg)',
                                      'Sex of the Child:Male':'Sex of child:Male',
                                      'HC':'HC (cm)'})

# Add n to columns
shorttable = shorttable.set_axis(['MAM (n=159)', 'Well-nourished (n=75)', 'pval'], axis=1)

# Final bit of formatting
shorttable.index = shorttable.index.str.replace(':', ' - ')
shorttable.index.name = ''
shorttable = shorttable.drop('Delivery Mode - Caesarean')
f.save(shorttable, 'shortpatientinfo')
