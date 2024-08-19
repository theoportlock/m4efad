#!/bin/bash
rm -r ../results/*
python format_metadata.py
python format_microbiome.py
python format_anthro.py
python format_bayleys.py
python format_psd.py
python format_wolkes.py
python format_lipids.py
