#!/usr/bin/env Rscript

# Load required packages
required_packages <- c("optparse", "vegan")

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
}

library(optparse)
library(vegan)

# Define command line arguments
option_list <- list(
  make_option(c("-d", "--distance_matrix"), type = "character", default = NULL,
              help = "Path to the distance matrix file", metavar = "character"),
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
opt$distance_matrix <- '../results/melonDist.tsv'
opt$metadata <- '../results/meta.tsv'
opt$fixed_effects <- 'Condition,Delivery_Mode,Sex_of_the_Child,Place_of_birth'

# Check if required arguments are provided
if (is.null(opt$distance_matrix) || is.null(opt$metadata) || is.null(opt$fixed_effects)) {
  stop("Please provide the paths for the distance matrix file, metadata file, and fixed effects.")
}

# Load data
distance_matrix <- tryCatch({
  read.csv(opt$distance_matrix, header = TRUE, sep = "\t", row.names = 1)
}, error = function(e) {
  stop("Error reading the distance matrix file: ", e$message)
})

metadata <- tryCatch({
  read.csv(opt$metadata, header = TRUE, sep = "\t", row.names = 1, check.names = FALSE)
}, error = function(e) {
  stop("Error reading the metadata file: ", e$message)
})

# Filter distance matrix by metadata indexes
common_samples <- intersect(rownames(distance_matrix), rownames(metadata))
distance_matrix <- distance_matrix[common_samples, common_samples]
metadata <- metadata[common_samples,]

# Convert factors to the appropriate format
fixed_effects <- strsplit(opt$fixed_effects, ",")[[1]]
fixed_effects <- paste0("`", fixed_effects, "`", collapse = "*")

# Create the formula for adonis2
full_formula <- as.formula(paste("distance_matrix ~", fixed_effects))

# Convert distance matrix to dist object
distance_matrix_dist <- as.dist(distance_matrix)

# Run adonis2
adonis2_result <- adonis2(full_formula, data = metadata, strata = metadata$ID)

# Print the results
print(adonis2_result)

# Run betadisper
#bd <- betadisper(distance_matrix_dist, metadata$Condition)
#anova_result <- anova(bd)
#perm_test <- permutest(bd, permutations = 99, pairwise = TRUE)

#print(anova_result)
#print(perm_test)

#pstat <- permustats(perm_test)
#densityplot(pstat, scales = list(x = list(relation = "free")))
#qqmath(pstat, scales = list(relation = "free"))

output_path <- opt$output
write.table(as.data.frame(adonis2_result), file = output_path, sep = "\t", quote = FALSE, col.names = NA)
