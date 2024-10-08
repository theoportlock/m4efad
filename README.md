# M4EFaD - Linking the Gut Microbiome to Neurocognitive Development in Bangladesh Malnourished Infants 
This repository is a store of code used for the production of the figures that were part of the study entitled: 'Linking the Gut Microbiome to Neurocognitive Development in Bangladesh Malnourished Infants'.

Shotgun metagenomics sequencing data filtered for human reads can be found on the NCBI-SRA: PRJNA1087376

These reads data were converted into species and functional profiles according to the Terra Worflow files in 'biobakery\_analysis/'

These profiles in addition to the additional lipid, EEG, Environmental factor, Bayley, and Wolkes datasets are found in the 'Data' directory which is available upon requrest and on the manuscript's figshare page.

## Installation

Python scripts are to be executed using python 3.8.17.

Requirements for python package installation are detailed in 'requirements.txt' and should take around 5 minutes to install with the command below. It is recommended that you do this in a python virual environment. This can be done (on mac or linux) with the commands:

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd metatoolkit/
pip install .
cd ../
```

## Running

Downstream analysis was done using scripts found in the 'code/' directory.

Additional scripts are located in the 'metatoolkit' package

The codebase is organised as per the figures in the manuscript.

Important: The 'data/' directory on this manuscripts figshare page (access token given to reviewers) should be downloaded and placed in this project's top directory.

All scripts for analysis should be ran from within the 'code/' directory under 'analysis.sh'

Please don't hesitate to reach out if you have any queries.

Theo.
