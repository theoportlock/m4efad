#!/usr/bin/env Rscript

# Load required packages
required_packages <- c("optparse", "vegan")

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
}

library(optparse)
library(vegan)
library(svglite)

# Define command line arguments
option_list <- list(
  make_option(c("-t", "--taxonomic_data"), type = "character", default = NULL,
              help = "Path to the taxo matrix file", metavar = "character"),
  make_option(c("-m", "--metadata"), type = "character", default = NULL,
              help = "Path to the metadata file", metavar = "character"),
  make_option(c("-f", "--fixed_effects"), type = "character", default = NULL,
              help = "Fixed effects (comma-separated, e.g., 'Factor1,Factor2')", metavar = "character"),
  make_option(c("-o", "--output"), type = "character", default = "../results/adonis2_results.tsv",
              help = "Path to the output TSV file", metavar = "character")
)

# Parse command line arguments
opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

opt$metadata = '../results/metaonehotstandard.tsv'
#opt$metadata = '../results/metaonehot.tsv'
#opt$metadata = '../results/meta.tsv'
opt$taxo = '../results/species.tsv'
#opt$taxo = '../results/pathways.tsv'
#opt$taxo = '../results/psd.tsv'
opt$fixed_effects = 'Condition.MAM,Delivery_Mode.Caesarean,Sex_of_the_Child.Female,Duration_of_Exclusive_Breast_Feeding_Months'

# Check if required arguments are provided
if (is.null(opt$taxo) || is.null(opt$metadata) || is.null(opt$fixed_effects)) {
  stop("Please provide the paths for the taxo matrix file, metadata file, and fixed effects.")
}

# Load data
taxo <- tryCatch({
  read.csv(opt$taxo, header = TRUE, sep = "\t", row.names = 1)
}, error = function(e) {
  stop("Error reading the taxo matrix file: ", e$message)
})

metadata <- tryCatch({
  read.csv(opt$metadata, header = TRUE, sep = "\t", row.names = 1, check.names = FALSE)
}, error = function(e) {
  stop("Error reading the metadata file: ", e$message)
})

# Filter taxo matrix by metadata indexes
common_samples <- intersect(rownames(taxo), rownames(metadata))
#taxo <- taxo[common_samples, common_samples]
taxo <- taxo[common_samples,]
metadata <- metadata[common_samples,]

# Convert factors to the appropriate format
fixed_effects <- strsplit(opt$fixed_effects, ",")[[1]]
#fixed_effects <- paste0("`", fixed_effects, "`", collapse = "*")
fixed_effects <- paste0("`", fixed_effects, "`", collapse = "+")

# Create the formula
full_formula <- as.formula(paste("taxo ~", fixed_effects))

# RDA
#output <- rda(taxo ~ ., data=metadata)
output <- rda(full_formula, data=metadata)

# Test the significance of each term
anova_terms <- anova(output, by = "term")

# Plot
svglite("../results/ordiplot1.svg", height=5,width=5)
ordiplot(output, scaling = 1, type = "text", cex=0.5)
#ordiplot(output, scaling = 1)
dev.off()
svglite("../results/ordiplot2.svg", height=5,width=5)
ordiplot(output, scaling = 2, type = "text")
#ordiplot(output, scaling = 2)
dev.off()

svglite("../results/summaryordiplot.svg", height=5,width=5)
ordiplot(output, scaling = 2)
#ordiplot(output, scaling = 2)
dev.off()


# Stats
RsquareAdj(output)
vif.cca(output) # variance inflation factor (<10 OK)
anova.cca(output, permutations = 1000) # full model
anova.cca(output, permutations = 1000, by="margin") # per variable


