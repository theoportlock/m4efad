#!/home/theop/venv/bin/python3
# -*- coding: utf-8 -*-

import argparse
import functions as f
import pandas as pd

parser = argparse.ArgumentParser(description='Calculate - compute a value for each sample based on features')
parser.add_argument('analysis')
parser.add_argument('subject')
known, unknown = parser.parse_known_args()
known = {k: v for k, v in vars(known).items() if v is not None}
unknown = eval(unknown[0]) if unknown != [] else {}

subject = known.get("subject")
analysis = known.get('analysis')
df = f.load(subject)

output = f.scale(analysis, df)
print(output)
f.save(output, subject+analysis)
