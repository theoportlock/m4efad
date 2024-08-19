#!/usr/bin/env python
import metatoolkit.functions as f
import pandas as pd

# load and format data
shaps = f.load('shap_interactsmean').reset_index()
shaps = shaps.set_axis(['source','target','shap_interactsmean'], axis=1)

regex = r'[\[\]\;\/\|\(\)\:\-\ ]'
shaps.source= shaps.source.str.replace(regex,'.', regex=True)
shaps.target= shaps.target.str.replace(regex,'.', regex=True)
shaps.set_index(['source','target'], inplace=True)

f.save(shaps, 'shap_interactsmeanformat')
