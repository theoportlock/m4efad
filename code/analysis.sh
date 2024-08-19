#!/bin/bash
####################################
###   MAM Baseline Manuscript
###   Code author: Theo Portlock
####################################

###### FORMAT RAW DATA ######
yes | rm -r ../results/*
python format_metadata.py
python format_microbiome.py
python format_anthro.py
python format_bayleys.py
python format_psd.py
python format_wolkes.py
python format_lipids.py

###### TABLE 1 - COVARIATES ######
calculate.py onehot meta
merge.py metaonehot anthro
filter.py -dt 'number' metaonehotanthro -o numeric
filter.py -dt 'bool' metaonehotanthro -o categories
fisher.py categories
change.py numeric --df2 categories
corr.py numeric
./merge_stats.py
./filt_malnutrition_stats.py
filter.py covariateedges -q 'qval < 0.05 and effect != inf and effect != -inf'
filter.py covariateedgesfilter -q 'effect > 0'
leiden_clustering.py covariateedgesfilterfilter -c effect

###### FIGURE 1 - MICROBIOME ######
./plot_anthro.py
taxo_summary.py taxo
covariates='Condition.MAM,Delivery_Mode.Caesarean,Sex_of_the_Child.Male,Duration_of_Exclusive_Breast_Feeding_Months'
Maaslin2.R -f $covariates ../results/taxo.tsv ../results/metaonehot.tsv ../results/taxochange
yes | cp ../results/taxochange/all_results.tsv ../results/taxochange.tsv
filter.py taxochange -q 'metadata == "Condition.MAM"'
volcano.py taxochangefilter --change coef --sig qval --fc 0.01 --pval 0.25
sig_summary.py taxochangefilter
Maaslin2.R -f $covariates ../results/pathwaysstrat.tsv ../results/metaonehot.tsv ../results/pathwaysstratchange
yes | cp ../results/pathwaysstratchange/all_results.tsv ../results/pathwaysstratchange.tsv
Maaslin2.R -f $covariates -a 0.001 ../results/pathways.tsv ../results/metaonehot.tsv ../results/pathwayschange
yes | cp ../results/pathwayschange/all_results.tsv ../results/pathwayschange.tsv
filter.py pathwayschange -q 'metadata == "Condition.MAM"'
Maaslin2.R -f $covariates ../results/melon.tsv ../results/metaonehot.tsv ../results/melonchange
yes | cp ../results/melonchange/all_results.tsv ../results/melonchange.tsv
filter.py melonchange -q 'metadata == "Condition.MAM"'
filter.py taxo --colfilt 's__'
./calculate_diversity.sh taxofilter
./adonis.R -d ../results/beta_unweighted-unifrac.tsv -m ../results/metaonehot.tsv -f $covariates -o ../results/beta_unweighted-unifracAdonis.tsv
./adonis.R -d ../results/beta_weighted-unifrac.tsv -m ../results/metaonehot.tsv -f $covariates -o ../results/beta_weighted-unifracAdonis.tsv
./adonis.R -d ../results/beta_bray-curtis.tsv -m ../results/metaonehot.tsv -f $covariates -o ../results/beta_bray-curtisAdonis.tsv
change.py alpha_diversity --df2 categories
stratify.py alpha_diversity Condition
box.py alpha_diversityCondition -y diversity_shannon
./beta_compare.py
ls ../results/beta* | parallel pcoa.py {}
ls ../results/*Pcoa.tsv | parallel stratify.py {} Condition
ls ../results/*PcoaCondition.tsv | parallel spindle.py {}

###### FIGURE 2 - BRAIN ######
Maaslin2.R -f $covariates -n None -t None ../results/bayley.tsv ../results/metaonehot.tsv ../results/bayleychange
yes | cp ../results/bayleychange/all_results.tsv ../results/bayleychange.tsv
filter.py bayleychange -q 'metadata == "Condition.MAM"'
sig_summary.py bayleychangefilter
Maaslin2.R -f $covariates -n None -t None ../results/wolkes.tsv ../results/metaonehot.tsv ../results/wolkeschange
yes | cp ../results/wolkeschange/all_results.tsv ../results/wolkeschange.tsv
filter.py wolkeschange -q 'metadata == "Condition.MAM"'
sig_summary.py wolkeschangefilter
Maaslin2.R -f $covariates -n None -t None -a '-5' ../results/psd.tsv ../results/metaonehot.tsv ../results/psdchange
yes | cp ../results/psdchange/all_results.tsv ../results/psdchange.tsv
filter.py psdchange -q 'metadata == "Condition.MAM"'
sig_summary.py psdchangefilter
./brain.py

###### FIGURE 3 - LIPIDS ######
Maaslin2.R -f $covariates -n None -t None ../results/lipids.tsv ../results/metaonehot.tsv ../results/lipidchange
yes | cp ../results/lipidchange/all_results.tsv ../results/lipidchange.tsv
filter.py lipidchange -q 'metadata == "Condition.MAM"'
sig_summary.py lipidchangefilter
volcano.py lipidchange --change coef --sig qval --fc 2 --pval 0.0000005
Maaslin2.R -f $covariates -n None -t None ../results/lipid_classes.tsv ../results/metaonehot.tsv ../results/lipid_classeschange
yes | cp ../results/lipid_classeschange/all_results.tsv ../results/lipid_classeschange.tsv
filter.py lipid_classeschange -q 'metadata == "Condition.MAM"'
sig_summary.py lipid_classeschangefilter
volcano.py lipid_classeschange --change coef --sig qval --fc 0.01 --pval 0.25
./lipids.py

###### FIGURE 4 - INTEGRATIVE ######
merge.py lipid_richness alpha_diversity
corr.py lipid_richnessalpha_diversity bayley
clustermap.py lipid_richnessalpha_diversitybayleycorr
datasets="taxo pathways melon lipid_classes psd wolkes bayley"
printf "$datasets" | parallel -d ' ' --jobs 1 'stratify.py {} Condition.MAM --df2 metaonehot'
printf "$datasets" | parallel -d ' ' --jobs 1 'predict.py classifier {}Condition.MAM'
paste $(printf "../results/%sCondition.MAMaucrocs.tsv " ${datasets}) > ../results/allaucrocs.tsv
merge.py $datasets -o alldata
./plot_aucrocs.py
filter.py alldata --prevail 0.1
stratify.py alldatafilter Condition.MAM --df2 metaonehot 
predict.py classifier alldatafilterCondition.MAM --shap_val --shap_interact -n 100
group.py alldatafilterCondition.MAMmeanabsshaps --func 'mean' --axis 'columns'
./shap_maaslin_compare.py
aucroc_curve.py alldatafilterCondition.MAMfpr_tpr
./plot_rel_shaps.py
group.py shap_interacts --func 'mean' --axis 'columns' --index_levels 2
./shap_interacts_reformat.py
filter.py shap_interactsmeanformat -q "abs(shap_interactsmean) > 0.0001" 
plot_circos.py shap_interactsmeanformatfilter shaps shap_interactsmean

###### FIGURE 5 - NETWORK ######
./network.py
./individual_plots.py

###### ASSEMBLE SUPP TABLES ######
makesupptable.py
