import pandas as pd
import metatoolkit.functions as f

df = f.load('../results/beta_bray-curtis.tsv')
species = f.filter(df, colfilt='s__')
species.columns = species.columns.str.replace('.*s__','', regex=True)

f.save(species, 'species')

