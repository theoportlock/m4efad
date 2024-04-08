# M4EFaD - Linking the Gut Microbiome to Neurocognitive Development in Bangladesh Malnourished Infants 
This repository is a store of code used for the production of the figures that were part of the study entitled: 'Linking the Gut Microbiome to Neurocognitive Development in Bangladesh Malnourished Infants'.

Shotgun metagenomics sequencing data filtered for human reads can be found on the NCBI-SRA: PRJNA1087376

These reads data were converted into species and functional profiles according to the Terra Worflow files in 'biobakery_analysis/'

These profiles in addition to the additional lipid, EEG, Environmental factor, Bayley, and Wolkes datasets are found in the 'Data' directory which is available upon requrest and on the manuscript's figshare page.

## Installation

All scripts are to be executed using python 3.8.17.

Requirements for python package installation are detailed in 'requirements.txt' and should take around 5 minutes to install with the command below. It is recommended that you do this in a python virual environment. This can be done (on mac or linux) with the commands:

```
python -m venv venv; source venv/bin/activate
pip install -r requirements.txt
```

## Running

Downstream analysis was done using scripts found in the 'code/' directory.

The codebase is organised as per the figures in the manuscript.

Scripts should be ran from within the 'code/' directory and in the order that follows.
```
python setup.py
python f1\_microbiome.py
python f2\_brain.py
python f3\_lipids.py
python f4\_AI.py
python f5\_network.py
python t1\_infantinfo.py
python s1\_explainedvariance.py
python makesupptable.py
```

Please don't hesitate to reach out if you have any questions.

Theo.
