#!/bin/bash
#####################################
#####   M4EFaD MAM Manuscript
#####   Code author: Theo Portlock
#####################################

#############################
###### FORMAT RAW DATA ######
#############################
yes | rm -r ../results/*
python format_metadata.py
python format_microbiome.py
python format_anthro.py
python format_bayleys.py
python format_psd.py
python format_wolkes.py
python format_lipids.py

##################################
###### TABLE 1 - COVARIATES ######
##################################
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

###################################
###### FIGURE 1 - MICROBIOME ######
###################################
./plot_anthro.py
taxo_summary.py taxo
./calculate_pb.py
covariates='Condition.MAM,Delivery_Mode.Caesarean,Sex_of_the_Child.Male,Duration_of_Exclusive_Breast_Feeding_Months'

# Species
Maaslin2.R -f $covariates ../results/taxo.tsv ../results/metaonehot.tsv ../results/taxochange
yes | cp ../results/taxochange/all_results.tsv ../results/taxochange.tsv
filter.py taxochange -q 'metadata == "Condition.MAM"'
volcano.py taxochangefilter --change coef --sig pval --fc 0.01 --pval 0.25
sig_summary.py taxochangefilter
./plot_species.py

# Pathways
Maaslin2.R -f $covariates ../results/pathwaysstrat.tsv ../results/metaonehot.tsv ../results/pathwaysstratchange
yes | cp ../results/pathwaysstratchange/all_results.tsv ../results/pathwaysstratchange.tsv
filter.py pathwaysstratchange -q 'metadata == "Condition.MAM"'
volcano.py pathwaysstratchangefilter --change coef --sig pval --fc 0.01 --pval 0.25
sig_summary.py pathwaysstratchangefilter

Maaslin2.R -f $covariates ../results/pathwaystaxo.tsv ../results/metaonehot.tsv ../results/pathwaystaxochange
yes | cp ../results/pathwaystaxochange/all_results.tsv ../results/pathwaystaxochange.tsv
filter.py pathwaystaxochange -q 'metadata == "Condition.MAM"'
volcano.py pathwaystaxochangefilter --change coef --sig pval --fc 0.01 --pval 0.25
sig_summary.py pathwaystaxochangefilter
stratify.py pathwaystaxo Condition
box.py pathwaystaxoCondition -y 'Catechol-Degradation'
box.py pathwaystaxoCondition -y 'Formaldehyde-Oxidation'

Maaslin2.R -f $covariates ../results/pathwayscomplete.tsv ../results/metaonehot.tsv ../results/pathwayscompletechange
yes | cp ../results/pathwayscompletechange/all_results.tsv ../results/pathwayscompletechange.tsv
filter.py pathwayscompletechange -q 'metadata == "Condition.MAM"'
volcano.py pathwayscompletechangefilter --change coef --sig pval --fc 0.01 --pval 0.05
sig_summary.py pathwayscompletechangefilter
stratify.py pathwayscomplete Condition
box.py pathwayscompleteCondition -y 'HSERMETANA-PWY: L-methionine biosynthesis III'

Maaslin2.R -f $covariates -a 0 ../results/pathways.tsv ../results/metaonehot.tsv ../results/pathwayschange
yes | cp ../results/pathwayschange/all_results.tsv ../results/pathwayschange.tsv
filter.py pathwayschange -q 'metadata == "Condition.MAM"'
volcano.py pathwayschangefilter --change coef --sig pval --fc 0.01 --pval 0.05

Maaslin2.R -f $covariates ../results/melon.tsv ../results/metaonehot.tsv ../results/melonchange
yes | cp ../results/melonchange/all_results.tsv ../results/melonchange.tsv
filter.py melonchange -q 'metadata == "Condition.MAM"'
volcano.py melonchangefilter --change coef --sig pval --fc 0.01 --pval 0.2

stratify.py melon Condition
box.py melonCondition -y 'nicotinic.acid'

# Diversity stats
filter.py taxo --colfilt 's__'
./calculate_diversity.sh taxofilter
./adonis.R -d ../results/beta_unweighted-unifrac.tsv -m ../results/metaonehot.tsv -f $covariates -o ../results/beta_unweighted-unifracAdonis.tsv
./adonis.R -d ../results/beta_weighted-unifrac.tsv -m ../results/metaonehot.tsv -f $covariates -o ../results/beta_weighted-unifracAdonis.tsv
./adonis.R -d ../results/beta_bray-curtis.tsv -m ../results/metaonehot.tsv -f $covariates -o ../results/beta_bray-curtisAdonis.tsv
scale.py standard metaonehot
./rda.R -t ../results/taxo.tsv -m ../results/metaonehotstandard.tsv -f $covariates 
calculate.py diversity taxo
change.py taxodiversity --df2 categories
stratify.py alpha_diversity Condition
box.py alpha_diversityCondition -y diversity_shannon
./beta_compare.py
ls ../results/beta* | parallel pcoa.py {}
ls ../results/*Pcoa.tsv | parallel stratify.py {} Condition
ls ../results/*PcoaCondition.tsv | parallel spindle.py {}

##############################
###### FIGURE 2 - BRAIN ######
##############################
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

./brain_plots.py

###############################
###### FIGURE 3 - LIPIDS ######
###############################
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

Maaslin2.R -f $covariates -n None -t None ../results/extralipids.tsv ../results/metaonehot.tsv ../results/extralipidchange
yes | cp ../results/extralipidchange/all_results.tsv ../results/extralipidchange.tsv
filter.py extralipidchange -q 'metadata == "Condition.MAM"'
sig_summary.py extralipidchangefilter
volcano.py extralipidchange --change coef --sig qval --fc 2 --pval 0.0000005

./lipid_plots.py

####################################
###### FIGURE 4 - INTEGRATIVE ######
####################################
./cluster_EV.py
filter.py taxo --colfilt 's__' -o species
merge.py lipid_richness alpha_diversity
corr.py lipid_richnessalpha_diversity bayley
clustermap.py lipid_richnessalpha_diversitybayleycorr
#datasets="species pathways melon lipid_classes psd wolkes bayley"
datasets="species pathways lipid_classes psd wolkes bayley"
printf "$datasets" | parallel -d ' ' --jobs 1 'stratify.py {} Condition.MAM --df2 metaonehot'
printf "$datasets" | parallel -d ' ' --jobs 1 'predict.py classifier {}Condition.MAM'
paste $(printf "../results/%sCondition.MAMaucrocs.tsv " ${datasets}) > ../results/allaucrocs.tsv
merge.py $datasets -o alldata
./plot_aucrocs.py
filter.py alldata --prevail 0.1
stratify.py alldatafilter Condition.MAM --df2 metaonehot 
./find_hyperparameters.py ../results/alldatafilterCondition.MAM
predict.py classifier alldatafilterCondition.MAM --shap_val --shap_interact -n 100
group.py alldatafilterCondition.MAMmeanabsshaps --func 'mean' --axis 'columns'
./shap_maaslin_compare.py
./plot_rel_shaps.py
group.py shap_interacts --func 'mean' --axis 'columns' --index_levels 2
./shap_interacts_reformat.py
./plot_shaps_thresh.py
filter.py shap_interactsmeanformat -q "abs(shap_interactsmean) > 0.00012" 
plot_circos.py shap_interactsmeanformatfilter shaps shap_interactsmean

###### FIGURE 5 - NETWORK ######
./create_edges.py
./individual_plots.py

###### ASSEMBLE SUPP TABLES ######
makesupptable.py
