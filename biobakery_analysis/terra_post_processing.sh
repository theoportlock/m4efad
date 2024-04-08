# Activate metaphlan3 conda environment to merge individual metaphlan3 profiles into one file
cd eresearch/m4efad/mgx/metaphlan3

conda activate metaphlan3

merge_metaphlan_tables.py ind_profiles/*.tsv > metaphlan3_taxonomy.tsv

conda deactivate

# Activate humann2 conda environment to merge individual humann3 output files into one file (humann2 conda also works for humann3 post-processing)
cd eresearch/m4efad/mgx/humann3

conda activate humann2

humann2_join_tables -i genefam -o humann3_genefam_rpk.tsv
humann2_join_tables -i pathway -o  humann3_pathway_rpk.tsv
humann2_join_tables -i pathway_coverage -o  humann3_pathway_coverage.tsv

# Renormalise counts from RPK (reads per kilobase) to CPM (copies per million)
humann2_renorm_table -i humann3_genefam_rpk.tsv -u "cpm" -o humann3_genefam_cpm.tsv
humann2_renorm_table -i humann3_pathway_rpk.tsv -u "cpm" -o humann3_pathway_cpm.tsv

# Split tables into community totals and species-stratfied versions
humann2_split_stratified_table -i humann3_genefam_cpm.tsv -o split/
humann2_split_stratified_table -i humann3_pathway_cpm.tsv -o split/

conda deactivate
