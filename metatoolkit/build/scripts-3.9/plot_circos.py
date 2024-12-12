#!/home/theop/venv/bin/python3
# -*- coding: utf-8 -*-

from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import functions as f
import numpy as np
import os
import pandas as pd
import sys
from pycircos import pycircos

def circos(edgelist, labels, weight='weight'):
    data = edgelist.reset_index()
    labels = labels.loc[(labels.index.isin(data.source)) | (labels.index.isin(data.target))]
    labels['ID'] = range(labels.shape[0])
    data = data.set_index('source').join(labels[['ID','dataset']], how='inner').rename(columns={'dataset':'source_dataset', 'ID':'source_ID'}).reset_index()
    data = data.set_index('target').join(labels[['ID','dataset']], how='inner').rename(columns={'dataset':'target_dataset', 'ID':'target_ID'}).reset_index()
    data = data.loc[data.source_dataset != data.target_dataset]
    Gcircle = pycircos.Gcircle
    Garc = pycircos.Garc
    circle = Gcircle()
    for i, row in labels.groupby('dataset', sort=False):
        arc = Garc(arc_id=i,
                   size=row.ID.max(),
                   #size=row.shape[0],
                   interspace=20,
                   label_visible=True)
        circle.add_garc(arc)
    circle.set_garcs()
    for i, row in data.iterrows():
        circle.chord_plot(
                start_list=(row['source_dataset'],
                            row['source_ID']-1,
                            row['source_ID'],
                            500),
                end_list=(row['target_dataset'],
                          row['target_ID']-1,
                          row['target_ID'],
                          500),
                facecolor=plt.cm.get_cmap('coolwarm')(row[weight]),
                ) 
    return circle

def parse_args(args):
    parser = argparse.ArgumentParser(
       prog='plot_circos.py',
       description='Script for plotting circos plots with edgelists'
    )
    parser.add_argument('subject', type=str, help='Data name or full filepath')
    parser.add_argument('labels', type=str, help='Data name or full filepath')
    parser.add_argument('weight', type=str, help='Column of weight function')
    return parser.parse_args(args)

arguments = ['../../fellowship/m4efad/baseline/results/shap_interactsmeanformatfilter.tsv',
             '../../fellowship/m4efad/baseline/results/shaps.tsv',
             'shap_interactsmean']
arguments = sys.argv[1:]
args = parse_args(arguments)

#edgelist = f.load(args.subject)
subject = args.subject
edgelist = f.load(subject)
labels = f.load(args.labels)
weight = args.weight

if os.path.isfile(args.subject):
    subject = Path(args.subject).stem
else:
    subject = args.subject

circos(edgelist, labels, weight=weight)
f.savefig(f'{subject}circos')
