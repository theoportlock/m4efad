import pandas as pd
import functions as f
import sys

from pycaret.classification import *

data = f.load('alldata')
meta = f.load('categories')

data = data.join(meta['Condition.MAM'])

setup(data, target = 'Condition.MAM', session_id = 123)

#best = compare_models()
#print(best)

model = create_model('rf')

best = tune_model(model)

#evaluate_model(best)

#f.setupplot(figsize=(2.5,2.5))
plot_model(best, plot = 'auc')

df = predict_model(best)

save_model(best, 'my_best_pipeline')

