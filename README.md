# M4EFaD: Linking the Gut Microbiome to Neurocognitive Development in Bangladesh Malnourished Infants  

This repository contains the code used to produce the figures for the study entitled: *"From Gut to Brain: Inter-connecting the Impacts of Moderate Acute Malnutrition on Child Cognitive Development"*  

### Data Availability  
- **Shotgun Metagenomics Data:**  
  Filtered human reads are available on the NCBI-SRA under project ID: **PRJNA1087376**.  
- **Species and Functional Profiles:**  
  Generated from the raw reads using workflows provided in the `biobakery_analysis/` directory.  
- **Lipidomics Data:**  
  Lipidomics data are available on the Metabolights database under project ID: **MTBLS10066**.  
- **Supplementary Datasets:**  
  The following additional datasets are available:  
  - Lipid profiles  
  - EEG data  
  - Environmental factors  
  - Bayley and Wolkes scores  
  These datasets are stored in the on the manuscript's Figshare page: **[DOI: 10.17608/k6.auckland.25560768](https://doi.org/10.17608/k6.auckland.25560768)**.  

---

## Installation  

The scripts require **Python 3.8.17**.  

### Setting Up the Environment  
To install the required packages, it is recommended to use a virtual environment. Follow these steps (Mac/Linux):  
1. Create and activate a virtual environment:  
   ```bash  
   python -m venv venv  
   source venv/bin/activate  
   ```  
2. Install the required Python packages:  
   ```bash  
   pip install -r requirements.txt  
   ```  
3. Install the `metatoolkit` package in this repository:  
   ```bash  
   pip install metatoolkit/
   ```  
4. Install Maaslin2 v1.73 as per the github instructions

---

## Running the Code  

### Directory Structure  
- **Code:**  
  The scripts for analysis are located in the `code/` directory and are organized according to the figures in the manuscript.  
- **Metatoolkit Package:**  
  Additional reusable scripts and functions are provided in the `metatoolkit` package.  
- **Results Directory:**  
  Download the data from the manuscript's Figshare page (**DOI: 10.17608/k6.auckland.25560768**) and place it in the project root directory in the results directory.  

### Execution Instructions  
1. Ensure the `results/` directory is in the project root with the data from the figshare page.  
2. Navigate to the `code/` directory
3. Run all commands following the execution order defined in `analysis.sh`

---

## Contact  

For questions or clarifications, feel free to reach out to theo.portlock@auckland.ac.nz

**Theo**  
