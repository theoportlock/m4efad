#!/bin/bash
# Biobakery's calculate diversity functions in their github/utils directory

metaphlan_file=$1
mkdir -p ../results/mbiome_alpha

# Transpose taxo file
transpose.py $metaphlan_file

# For alpha diversity measures
for metric in gini shannon richness simpson
do
calculate_diversity.R \
	-f ../results/${metaphlan_file}_T.tsv \
	-t ../data/mpa_v31_CHOCOPhlAn_201901_species_tree.nwk \
	-d alpha \
	-p alpha \
	-m $metric \
	-s s__ \
	-o ../results/mbiome_alpha
done

# Combine alpha diversity metrics and cleanup
merge.py ../results/mbiome_alpha/* -o alpha_diversity &&
rm -r ../results/mbiome_alpha

# For beta diversity measures
for metric in bray-curtis jaccard weighted-unifrac unweighted-unifrac clr aitchison
do
calculate_diversity.R \
	-f ../results/${metaphlan_file}_T.tsv \
	-t ../data/mpa_v31_CHOCOPhlAn_201901_species_tree.nwk \
	-d beta \
	-p beta \
	-m $metric \
	-s s__ \
	-o ../results
done

# Format clr data
python -c "import pandas as pd; pd.read_csv(\"../results/beta_clr.tsv\", index_col=0, sep=\"\t\").T.to_csv(\"../results/beta_clr.tsv\", sep=\"\t\")"

# Cleanup
rm ../results/beta_species.log
rm ../results/${metaphlan_file}_T.tsv 

