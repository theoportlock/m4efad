#!/usr/bin/env python
import metatoolkit.functions as f

meta = f.load('categories')
df = f.load('taxo')

df.columns = df.columns.str[3:]
df = f.filter(df, prevail=0.1)
df = f.mult(df)
pb = f.calculate("pbratio", df)
f.save(pb, 'PBratio')

ch = f.change(pb, meta)
fch = f.filter(ch.reset_index().set_index('source'), query='MWW_qval < 0.05')
print(fch)

