#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import functions as f
import numpy as np
import os
import pandas as pd
import sys

def aucroc(df):
    f.setupplot()
    meanroc = df.groupby(level=0).mean()
    stdroc = df.groupby(level=0).std()
    fig, ax= plt.subplots()
    ax.plot(df.fpr, df.tpr, alpha=0.4)
    ax.plot(meanroc.fpr, meanroc.tpr)
    ax.fill_between(meanroc.fpr, meanroc.tpr-stdroc.tpr, meanroc.tpr+stdroc.tpr, alpha=0.2)
    ax.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc="lower right")
    return ax

def parse_args(args):
    parser = argparse.ArgumentParser(
       prog='aucroc_curve.py',
       description='Script for plotting multiple AUCROC curves'
    )
    parser.add_argument('subject', type=str, help='Data name or full filepath')
    return parser.parse_args(args)

arguments = sys.argv[1:]
args = parse_args(arguments)

df = f.load(args.subject)

if os.path.isfile(args.subject):
    subject = Path(args.subject).stem
else:
    subject = args.subject

aucroc(df)
f.savefig(f'{subject}aucroc_curve')
