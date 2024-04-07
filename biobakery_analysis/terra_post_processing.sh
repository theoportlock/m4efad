# Activate metaphlan3 conda environment to merge individual metaphlan3 profiles into one file
cd eresearch/gutbugs/mgx/metaphlan3

conda activate metaphlan3

merge_metaphlan_tables.py ind_profiles/*.tsv > gutbugs_metaphlan3_taxonomy.tsv

conda deactivate

# Activate humann2 conda environment to merge individual humann3 output files into one file (humann2 conda also works for humann3 post-processing)
cd eresearch/gutbugs/mgx/humann3

conda activate humann2

humann2_join_tables -i genefam -o gutbugs_humann3_genefam_rpk.tsv
humann2_join_tables -i pathway -o  gutbugs_humann3_pathway_rpk.tsv
humann2_join_tables -i pathway_coverage -o  gutbugs_humann3_pathway_coverage.tsv

# Renormalise counts from RPK (reads per kilobase) to CPM (copies per million)
humann2_renorm_table -i gutbugs_humann3_genefam_rpk.tsv -u "cpm" -o gutbugs_humann3_genefam_cpm.tsv
humann2_renorm_table -i gutbugs_humann3_pathway_rpk.tsv -u "cpm" -o gutbugs_humann3_pathway_cpm.tsv

# Split tables into community totals and species-stratfied versions
humann2_split_stratified_table -i gutbugs_humann3_genefam_cpm.tsv -o split/
humann2_split_stratified_table -i gutbugs_humann3_pathway_cpm.tsv -o split/

conda deactivate

# Strainphlan
for f in *.gz; do tar -xvf $f; done

mkdir alignments

mv s__*/&.aln alignments/

tar -cf gutbugs_strainphlan3_alignments.tar alignments/*.aln
gzip gutbugs_strainphlan3_alignments.tar