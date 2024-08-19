# Characteristics
Infants with MAM (N = 140) and well-nourished controls (N = 70)
EXPAND ON RECRUITMENT

MAM was found to associate with several confounding sociodemographic factors: TableS1
 - principle toilet system used (septic-tank/toilet)
 - water treatment method (boil)
 - toilet facility (shared with other households)
 - length of time lived in current household
 - mother’s income
 - years of father education
 - father’s education level
 - monthly total expenditure
 - mother’s occupation (housewife)
COVARIATES AND NETWORK?

# Microbiome
Stool metagenomes were extracted, sequenced (40.53 +- 8.5 million reads)

Across all samples:
3 kingdoms
17 phyla
31 classes
51 orders
100 families
226 genera
749 species
611 functional pathways
2,828,874 gene families

For all children, there was a mean species richness of 50.3 ± 16.4 per sample and mean Shannon diversity of 2.96 ± 0.72.
IS THIS DIFFERENT TO NORMAL 1YR OLDS?

MAM was associated with a lower Shannon diversity (p = 0.025) and Pielou’s evenness (p = 0.009) than their well-nourished counterparts 
IS THIS OBSERVED PREVIOUSLY AND IS THIS CONTROLING FOR OTHER CONFOUNDERS?

The observed differences in alpha diversity were underscored by a significant compositional difference in the Bray-Curtis dissimilarity between the nutritional groups (PERMANOVA, R2 = 2.22, p = 0.008, Figure1d),

differential abundance of 7/115 species (6.1%) (Figure1e, TableS3).

MAM child gut microbiomes had a greater prevalence and abundance of five species including Prevotella copri (coef = 0.64, q = 0.490), Rothia mucilaginosa (coef = 2.47, q = 0.032), and Streptococcus salivarius (coef = 2.39, q = 0.032), compared to well-nourished controls (Figure1c).
Enrichment in these species was associated with the depletion and reduction in the prevalence of the sphingolipid-producing species Bacteroides fragilis (coef = -0.62, q = 0.490).

This putative reciprocal relationship contributed to an increase in the Prevotella to Bacteroides (P/B) ratio of the MAM children (coef = 2.80, p = 0.050) (Figure1f).

Functional pathway analyses revealed no significant differences in the composition of the overall functionome between MAM and well-nourished controls (PERMANOVA, R2 = 8.76, p = 0.365).

After false discovery rate adjustment, there were no significant differences in the relative abundances of individual pathways (TableS4).

# Brain
Increase in high-alpha (9-12 Hz), beta (12-30 Hz) and gamma (30-45 Hz) frequency bands distributed across occipital, temporal parietal, and frontal regions (Figure2a & b).
CHANGE THIS SECTION FOR READABILITY

When compared to well-nourished children, there was a significant reduction in Expressive Communication, Fine Motor, and Gross Motor scores in the MAM children (mean difference (MAM - well-nourished) = -2.02, -1.68, -2.69, p = 0.004, 0.001, 0.008, respectively; autorefFigure2c, autorefTableS6).

As with the Bayley scoring, vocalisation scores were reduced in MAM children (mean difference (MAM - well-nourished) = -1.47, p = 2.05e-10) in addition to corresponding reductions to activity and approach scores.
EMPHASIZE THAT BAYLEY IS WESTERN - READ OTHER DOCUMENT FOR PROTOCOL

# Blood
Therefore, untargetted LC-MS/MS was used to assign and quantify the levels of 792 plasma lipids in the children of the cohort (autorefFigure3a).

MAM was associated with major changes (254/792 - 32%) to the plasma lipidome.

Of these changes, 105 (13%) compounds increased and 149 (19%) decreased in concentration (autorefFigure3, autorefTableS8, autorefTableS9).

Enrichment in the relative abundance of three lipids with diverse functions was observed, including those that are known to be specific to neurological development and function such as ceramides (in particular the long chain ceramide Cer 41:2;O2 (coef = 2.236, q = 0.001).

By contrast, long chain sphingomyelins (SM 38:1;O2, coef = 0.505, q = 0.001)) and others were observed to increase in relative abundance in MAM children.

Several lysolipids from the lysophosphatidylcholine (LPC), and lysophosphatidylethanolamine (LPE) classes were enriched in well-nourished infant plasma.
STATS HERE

In addition, the commonly bacterially derived Odd-Chain Fatty Acids (OCFA) were shown to be enriched in well-nourished children.
STATS HERE

# AI
This analysis began with the evaluation of difference in beta diversity using PERMANOVA with respect to all phenotypical covariates including MAM (autorefFigureS1).
Across all covariates, nutritional status alone significantly explained the variance in the greatest number of datasets.
STATS HERE OR REMOVE
MULTIMODAL ANALYSIS RATHER THAN AI?

Within the predictors trained on individual feature sets, plasma lipids (AUCROC = 0.97) were the best predictor of MAM in 12-month-old children, followed by gut microbiome taxonomic and functioal profiles (AUCROC = 0.95, 0.93 respecively), then followed by  brain and behavioural metrics (i.e., Bayley score, EEG, and Wolke score AUCROC = 0.93, 0.88, 0.88).

The ensemble models were highly predictive of MAM (AUCROC = 0.82).
ERROR BARS

SHAP was highly correlated with -log(MWU significance) (Spearman correlation between mean SHAP score and MWU -log2(p) of rho = 0.74)).
WERE SIGNIFICANTLY CORRELATED FEATURES MORE LIKELY TO BE IMPORTANT

For example, these included MAM depleted Faecalibacterium prausnitzii (mean SHAP = 0.008), and Odoribacter splanchnicus (mean SHAP = 0.006) or MAM enriched Bifidobacterium breve (mean SHAP = 0.007), and Haemophilus parainfluenzae (mean SHAP = 0.007).
REMOVE THIS AND REPLACE WITH PREVIOUS COMMENT

# Networks
Spearman correlation of the features that were important in predicting MAM were calculated and filtered by significance (q < 0.05) (1052/3906 correlations, autorefTableS11).

Important features (i.e., mean absolute SHAP score > 0.002 (above 85%)) were more likely to be significantly correlated (q < 0.05) with one another (autorefFigure5a) than unimportant features (mean absolute SHAP < 0.002).

Plasma lipids that were enriched/depleted in the MAM condition were positively/negatively correlated with the anthropometric measures WLZ/WHZ, MUAC, and weight respectively.
VALUES

Cluster analyses revealed that those features which were different between MAM and well-nourished were positively correlated.
VALUES

A cluster of B. fragilis, pyruvate fermentation pathways, plasma ceramides, EEG PSD and Expressive Communication was identified as being highly correlated with the well-nourished state (autorefFigure5b).
CALCULATE CLUSTER

Plasma lipids that were depleted (q < 0.05, coef < 0) from the MAM infant samples were also negatively correlated with EEG PSD high-frequency amplitudes.

Notably, EEG measurements were also correlated with bacterial pyruvate fermentation pathways and B. fragilis relative abundance.

A cluster of P. copri, glycolysis, peptidoglycan biosynthesis, BCAA pathways, and plasma sphingomyelins was identified as being associated with the MAM condition. 
CALCULATE CLUSTER

