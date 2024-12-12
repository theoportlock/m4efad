#!/usr/bin/env Rscript
#
# Simple wrapper for vegan adonis function in R

# install required packages

required_pkg <- c("optparse", "ape", "rbiom", "compositions", "BiocManager")
a <- sapply(required_pkg, function(x) {  if (!requireNamespace(x, quietly = TRUE))
  install.packages(x, repos = "http://cran.us.r-project.org")
})
if (! "microbiome" %in% installed.packages()){
  BiocManager::install("microbiome")
}

# accept arguments from command line

library("optparse")

option_list = list(
  
  make_option(c("-f", "--file"), action="store", type="character", default=NULL, 
              help="Merged MetaPhlAn profiles. 
                A table with samples as columns and species as rows is required.",
              metavar="character"),
  
  make_option(c("-o", "--out_directory"), action="store", type="character", default="diversity_analysis",
              help="output directory.
                [default = %default]"),
  
  make_option(c("-p", "--outfile_prefix"), action="store", type="character", default=NULL,
              help="file name prefix of the output distance matrix and log files.
                [default = input file basename]"),
  
  make_option(c("-t", "--tree"), action="store", type="character", default=NULL, 
              help="Full path to the MetaPhlAn species Newick tree.
                Mandatory for computing UniFrac distances."),
  
  make_option(c("-d", "--diversity"), action="store", type="character", default="beta", 
              help="Choose whether to calculate alpha or beta diversity. 
                Options are alpha or beta.
                [default = %default]"),
  
  make_option(c("-m", "--metric"), action="store", type="character", default="bray-curtis", 
              help="Name of the function to use when calculating diversity.
                Options for alpha diversity are richness, shannon, simpson, gini.
                Options for beta diversity are bray-curtis, jaccard, weighted-unifrac, unweighted-unifrac, clr, aitchison.
                [default = %default]"),
  
  make_option(c("-s", "--taxon_separator"), action="store", type="character", default="t__", 
              help="taxon separator used in the input MetaPhlAn table.
                Options are: t__ for MetaPhlAn4 profiles and s__ for MetaPhlAn3 profiles.
                [default = %default]")
); 

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

if (is.null(opt$file)){
  print_help(opt_parser)
  stop('At least one argument must be supplied (input file).tsv', call.=FALSE)
}

if(! (opt$diversity %in% c('alpha', 'beta'))){
  write(paste0('Method "', opt$diversity, '" not available!'), stdout())
  write(paste0('Available diversity analyses are "alpha" and "beta"'), stdout())
  quit(status = -1)
}

if(opt$diversity =="alpha" & ! (opt$metric %in% c('richness', 'shannon', 'simpson', 'gini'))){
  write(paste0('Method "', opt$metric, '" not available for alpha diversity'), stdout())
  write(paste0('Available alpha-diversity metrics are "richness", shannon", "simpson", "gini".'), stdout())
  quit(status = -1)
}

if(opt$diversity =="beta" & ! (opt$metric %in% c('bray-curtis', 'jaccard', 'weighted-unifrac', 'unweighted-unifrac', 'clr', 'aitchison'))){
  write(paste0('Method "', opt$metric, '" not available for beta diversity'), stdout())
  write(paste0('Available beta-diversity distance functions are "bray-curtis", "jaccard", "weighted-unifrac", "unweighted-unifrac", "clr", "aitchison".'), stdout())
  quit(status = -1)
}

if(! (opt$taxon_separator %in% c('t__', 's__'))){
  write(paste0('Taxon separator "', opt$taxon_separator, '" is not available'), stdout())
  write(paste0('Possible taxon separators are "t__" for MetaPhlAn4 profiles and "s__" for MetaPhlAn3 profiles.'), stdout())
  quit(status = -1)
}

if(is.null(opt$tree) & grepl('unifrac', opt$metric)){
  write(paste0('Selected beta-diversity metric: "', opt$metric, '"'), stdout())
  stop("A  tree is mandatory for computing UniFrac distances. (input tree).nwk", call.=FALSE)
}

for(x in c(opt$file, opt$tree)){
  if(!file.exists(x)){
    stop(paste0('Input file "', x, '" does not exist!'), call.=FALSE)
  }
}

if(is.null(opt$outfile_prefix)){
  outfile_prefix <- basename(opt$file)
  outfile_prefix <- tools::file_path_sans_ext(outfile_prefix)
} else {
  outfile_prefix <- opt$outfile_prefix
}

# Install and load required packages
#install.packages("vegan")
library(vegan)

# Parse command-line arguments
args <- commandArgs(trailingOnly = TRUE)
#data_file <- '../results/sleepIDRecovery0.0vs1.0.tsv
data_file <- args[1]
group_variable <- args[2]
distance_variable <- args[3]
results_directory <- '../results/' + 'adonis'

# Load data from TSV file
data <- read.table(data_file, sep="\t", header=TRUE)

# Perform PERMANOVA analysis
permanova_result <- adonis(formula(paste(distance_variable, "~", group_variable)), data=data, permutations=999)

# Print the PERMANOVA results
print(permanova_result)

# Save results to specified directory
results_file <- paste(results_directory, "/permanova_results.txt", sep="")
write.table(permanova_result, file=results_file, quote=FALSE)

