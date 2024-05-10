if (!requireNamespace("BiocManager", quietly=TRUE))
  install.packages("BiocManager")
BiocManager::install("ChemmineR")

library("ChemmineR") # Loads the package

# Install and load required packages
install.packages("Chemminer")
library(Chemminer)

# Load SMILES files
