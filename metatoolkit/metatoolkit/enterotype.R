#!/bin/Rscript
BiocManager::install("DirichletMultinomial")
require(DirichletMultinomial)
require(dplyr)
require(parallel)

# Preamble
options(width=70, digits=2)
full <- FALSE
.qualitative <- DirichletMultinomial:::.qualitative
dev.off <- function(...) invisible(grDevices::dev.off(...))

# Load data
genusMat <- read.table("../results/genus.tsv", header=T, row.names=1, dec=".", sep="\t")
genusMat <- genusMat * 1e2

# DMN clustering
set.seed(1)
genusFit <- mclapply(1:7, dmn, count=as.matrix(genusMat), verbose=TRUE, mc.cores=16)
lplc <- sapply(genusFit, laplace)
pdf("../results/min-laplace.pdf")
plot(lplc, type="b", xlab="Number of Dirichlet Components" ,ylab="Model Fit")
dev.off()
best <- genusFit[[which.min(lplc)]]
k <- length(best@mixture$Weight)
p0 <- fitted(genusFit[[1]], scale=TRUE)
p3 <- fitted(best, scale=TRUE)
colnames(p3) <- paste("m", 1:k, sep="")
(meandiff <- colSums(abs(p3 - as.vector(p0))))
diff <- rowSums(abs(p3 - as.vector(p0)))
o <- order(diff, decreasing=TRUE)
cdiff <- cumsum(diff[o]) / sum(diff)
df <- head(cbind(Mean=p0[o], p3[o,], diff=diff[o], cdiff), 10)

### assigning cluster names ####
clusterAssigned = apply(best@group, 1, function(x) which.max(x))
clusterAssignedList = split(names(clusterAssigned), clusterAssigned)
out = stack(clusterAssignedList)
write.csv(out, '../results/enterotypes.csv', quote=F)
