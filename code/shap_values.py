#!/usr/bin/env python
# -*- coding: utf-8 -*-

from imblearn.over_sampling import SMOTE
from itertools import permutations
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, mean_absolute_error, r2_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse
import functions as f
import numpy as np
import os
import pandas as pd
import shap
import sys

def predict(df):
    model = RandomForestClassifier(n_jobs=-1,
                                   random_state=42,
                                   bootstrap=False,
                                   max_depth=None,
                                   min_samples_leaf=8,
                                   min_samples_split=20,
                                   n_estimators=400)
    X, y = df.reset_index(drop=True), df.index
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)
    y_prob = model.predict_proba(X)[:, 1]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_scaled)
    shap_values.feature_names = X.columns
    # Plot
    f.setupplot()
    shap.plots.beeswarm(shap_values[:,:,1], show=False)
    f.savefig('beeswarm')
    shap.plots.waterfall(shap_values[71,:,1], show=False)
    f.savefig('waterfall')
    shap_interaction_values = explainer.shap_interaction_values(X_scaled)[:,:,:,1]
    out = []
    for i, I in enumerate(shap_interaction_values):
        out.append(pd.DataFrame(I, index=X.columns, columns=X.columns).stack().to_frame(X.index[i]))
    outdf = pd.concat(out,axis=1)
    vals = ('Expressive Communication Score','Vocalisation')
    vals = ('Sterols [ST01]','Monoradylglycerols [GL01]')
    vals = ('s__Streptococcus_salivarius','Oxidized glycerophospholipids [GP20]')
    vals = ('s__Streptococcus_salivarius','s__Bacteroides_fragilis')
    vals = ('s__Faecalibacterium_prausnitzii','s__Bacteroides_fragilis')
    vals = ('s__Prevotella_copri','Oxidized glycerophospholipids [GP20]')
    vals = ('s__Bacteroides_fragilis','Oxidized glycerophospholipids [GP20]')
    vals = ('s__Bacteroides_fragilis','BRANCHED-CHAIN-AA-SYN-PWY: superpathway of branched chain amino acid biosynthesis')

    ffoutdf = outdf.loc[vals]
    fX = X
    odf = ffoutdf.to_frame('interaction').join(fX)
    #sns.scatterplot(odf, x=vals[0], y=vals[1], hue='interaction', palette='coolwarm', hue_norm=(-3e-4,3e-4), style=y)
    sns.scatterplot(odf, x=vals[0], y=vals[1], hue='interaction', palette='coolwarm', style=y)
    sns.scatterplot(odf, x='interaction', y=y_prob, palette='coolwarm', hue_norm=(-3e-4,3e-4), style=y)

    # Dig into fragilis
    fffoutdf = outdf.xs('s__Bacteroides_fragilis')
    fffoutdf.agg(['mean','std'], axis=1).sort_values('mean').tail(n=25)['mean'].plot.barh()

def parse_args(args):
    parser = argparse.ArgumentParser(
       prog='predict.py',
       description='Random Forest Classifier/Regressor with options'
    )
    parser.add_argument('subject', type=str, help='Data name or full filepath')
    return parser.parse_args(args)

arguments = ['alldatafilterCondition.MAM']
arguments = sys.argv[1:]
args = parse_args(arguments)

df = f.load(args.subject)
predict(df, args.analysis, shap_val=args.shap_val, shap_interact=args.shap_interact, n_iter=args.n_iter)
