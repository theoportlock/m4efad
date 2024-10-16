#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from sklearn.datasets import load_iris  # Just for example
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import train_test_split
import argparse
import metatoolkit.functions as f
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys

def tune_random_forest(df):
    # Define the random forest classifier
    X, y= df, df.index
    rf = RandomForestClassifier(random_state=41)

    # Define the hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200, 400],
        'max_depth': [None],
        'min_samples_split': [5, 10, 20],
        'min_samples_leaf': [4, 8, 16],
        'bootstrap': [False]
    }

    # GridSearchCV to search for best hyperparameters
    grid_search = GridSearchCV(
        estimator=rf, 
        param_grid=param_grid, 
        cv=10,
        n_jobs=-1,
        verbose=2,
        scoring='roc_auc_ovr'
    )

    # Perform grid search
    grid_search.fit(X, y)

    # Print best hyperparameters and best score
    print("Best hyperparameters:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)
    
    return pd.Series(grid_search.best_params_).to_frame('values')

def parse_args(args):
    parser = argparse.ArgumentParser(
       prog='find_hyperparameters.py',
       description=''' find_hyperparameters - 
       finds the best hyperparameters for RFC using
       10-fold cross validation
       '''
    )
    parser.add_argument('subject', type=str, help='Data name or full filepath')
    return parser.parse_args(args)

if __name__ == "__main__":
    arguments = sys.argv[1:]
    args = parse_args(arguments)

    if os.path.isfile(args.subject):
        subject = Path(args.subject).stem
    else:
        subject = args.subject

    df = f.load(subject)
    hyperparams = tune_random_forest(df)
    f.save(hyperparams, f'{subject}_hyperparams')
