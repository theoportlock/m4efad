import pandas as pd
import functions as f
import sys

from pycaret.classification import *

data = f.load('alldataCondition.MAM').reset_index()
#data = pd.read_csv(sys.argv[1], sep='\t', index_col=0)

setup(data, target = 'Condition.MAM', session_id = 123)

#best = compare_models()
#print(best)

model = create_model('rf')

best = tune_model(model)

evaluate_model(best)

#f.setupplot(figsize=(2.5,2.5))
plot_model(best, plot = 'auc')

predict_model(best)

save_model(best, 'my_best_pipeline')

