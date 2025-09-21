# ---- Packages ----
suppressPackageStartupMessages({
  library(proxy)   # For Jaccard and Simpson distances
})

# ---- Helper: Get script directory ----
get_script_dir <- function() {
  # When executed via Rscript
  args <- commandArgs(trailingOnly = FALSE)
  i <- grep("^--file=", args)
  if (length(i) > 0) {
    return(dirname(normalizePath(sub("^--file=", "", args[i]))))
  }
  # Fallback: current working directory
  normalizePath(getwd())
}

# ---- Helper: Resolve project root automatically ----
# Climb up until we find a directory that contains both "output" and "genomes"
resolve_project_root <- function(start_dir, max_up = 8) {
  p <- normalizePath(start_dir)
  for (k in seq_len(max_up)) {
    if (dir.exists(file.path(p, "output")) && dir.exists(file.path(p, "genomes"))) {
      return(p)
    }
    parent <- dirname(p)
    if (parent == p) break
    p <- parent
  }
  stop("Project root not found (looked for dirs: 'output' and 'genomes'). Start_dir=", start_dir)
}

# ---- Determine paths ----
SCRIPT_DIR <- get_script_dir()
ROOT <- resolve_project_root(SCRIPT_DIR)
cat(sprintf("[INFO] PROJECT ROOT: %s\n", ROOT))

INPUT_DIR  <- file.path(ROOT, "output", "02_gene_screening", "amr_and_vf_genes")
OUTPUT_DIR <- file.path(ROOT, "output", "03_distance", "genotype_distance", "genotypic_distance_matrix_all_phylogroups")
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# ---- Input files ----
amr_csv <- file.path(INPUT_DIR, "amr_genes_presence_absence.csv")
vf_csv  <- file.path(INPUT_DIR, "vf_genes_presence_absence.csv")

# Validate file existence
stopifnot(file.exists(amr_csv), file.exists(vf_csv))

# ---- Load binary presence/absence matrices ----
# Each row = genome, each column = gene, values = {0,1}
amr_binary <- read.csv(amr_csv, row.names = 1, check.names = FALSE)
vf_binary  <- read.csv(vf_csv,  row.names = 1, check.names = FALSE)


# Force numeric (important if some columns are read as characters)
amr_binary[] <- lapply(amr_binary, function(x) as.integer(as.character(x)))
vf_binary[]  <- lapply(vf_binary,  function(x) as.integer(as.character(x)))

# ---- Distance calculation ----
# Jaccard distance = 1 - (intersection / union)
amr_jaccard_matrix <- as.matrix(proxy::dist(amr_binary, method = "Jaccard"))
vf_jaccard_matrix  <- as.matrix(proxy::dist(vf_binary,  method = "Jaccard"))

# Simpson distance = 1 - overlap coefficient
amr_simpson_matrix <- as.matrix(proxy::dist(amr_binary, method = "Simpson"))
vf_simpson_matrix  <- as.matrix(proxy::dist(vf_binary,  method = "Simpson"))

# ---- Handle genomes with zero AMR genes ----
# For genomes with no AMR genes (all zeros), Simpson distance is undefined.
# Replace distances involving such genomes with 0.
amr_zero <- rowSums(amr_binary, na.rm = TRUE) == 0
if (any(amr_zero)) {
  amr_simpson_matrix[amr_zero, ] <- 0
  amr_simpson_matrix[, amr_zero] <- 0
}

# ---- Export distance matrices ----
out1 <- file.path(OUTPUT_DIR, "amr_jaccard_distance_matrix_all_genomes.csv")
out2 <- file.path(OUTPUT_DIR, "amr_simpson_distance_matrix_all_genomes.csv")
out3 <- file.path(OUTPUT_DIR, "vf_jaccard_distance_matrix_all_genomes.csv")
out4 <- file.path(OUTPUT_DIR, "vf_simpson_distance_matrix_all_genomes.csv")

write.csv(amr_jaccard_matrix,  out1)
write.csv(amr_simpson_matrix, out2)
write.csv(vf_jaccard_matrix,  out3)
write.csv(vf_simpson_matrix, out4)

cat("[OK] wrote:\n", out1, "\n", out2, "\n", out3, "\n", out4, "\n", sep = "")
