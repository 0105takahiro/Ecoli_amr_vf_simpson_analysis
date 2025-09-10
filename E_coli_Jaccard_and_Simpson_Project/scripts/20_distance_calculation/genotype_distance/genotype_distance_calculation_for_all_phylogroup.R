# Load required package
library(proxy)

# ---- Config: project name to lock as ROOT ----
PROJECT_NAME <- "E_coli_Jaccard_and_Simpson_Project"

# ---- Get this script's absolute path ----
get_script_path <- function() {
  # When executed via Rscript:
  args <- commandArgs(trailingOnly = FALSE)
  i <- grep("^--file=", args)
  if (length(i) > 0) {
    return(normalizePath(sub("^--file=", "", args[i])))
  }
  # When sourced in an interactive session:
  if (!is.null(sys.frames()[[1]]$ofile)) {
    return(normalizePath(sys.frames()[[1]]$ofile))
  }
  # (Optional) RStudio fallback
  p <- tryCatch({
    if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
      normalizePath(rstudioapi::getSourceEditorContext()$path)
    } else ""
  }, error = function(e) "")
  if (nzchar(p)) return(p)

  stop("Unable to determine script path. Run via Rscript or source this file directly.")
}

# ---- Strict ROOT resolver: climb until folder name == PROJECT_NAME ----
resolve_project_root <- function(script_path) {
  p <- dirname(script_path)
  repeat {
    if (basename(p) == PROJECT_NAME) return(p)
    parent <- dirname(p)
    if (parent == p) break  # reached filesystem root
    p <- parent
  }
  stop(sprintf("Project root '%s' not found above: %s", PROJECT_NAME, script_path))
}

# ---- Use it ----
SCRIPT_PATH <- get_script_path()
ROOT <- resolve_project_root(SCRIPT_PATH)
cat(sprintf("[INFO] PROJECT ROOT: %s\n", ROOT))

# Example: define IO paths under ROOT
INPUT_DIR  <- file.path(ROOT, "output", "amr_and_vf_genes")
OUTPUT_DIR <- file.path(ROOT, "output", "genotypic_distance_matrix_all_phylogroups")
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# ----------------------------
# Load binary presence/absence matrices for AMR and VF genes
# ----------------------------
# Each row represents a genome, each column represents a gene.
# Values: 1 = gene present, 0 = gene absent
amr_binary <- read.csv(file.path(INPUT_DIR, "amr_genes_presence_absence.csv"), 
                       row.names = 1, check.names = FALSE)
vf_binary  <- read.csv(file.path(INPUT_DIR, "vf_genes_presence_absence.csv"),  
                       row.names = 1, check.names = FALSE)

# ----------------------------
# Calculate pairwise Jaccard distances
# ----------------------------
# Jaccard distance = 1 - (intersection / union) of two sets
amr_jaccard_matrix <- as.matrix(proxy::dist(amr_binary, method = "Jaccard"))
vf_jaccard_matrix  <- as.matrix(proxy::dist(vf_binary,  method = "Jaccard"))

# ----------------------------
# Calculate pairwise Simpson distances
# ----------------------------
# Simpson distance is defined as 1 - overlap coefficient
# Overlap coefficient = |A ∩ B| / min(|A|, |B|)
amr_simpson_matrix <- as.matrix(proxy::dist(amr_binary, method = "Simpson"))
vf_simpson_matrix  <- as.matrix(proxy::dist(vf_binary,  method = "Simpson"))

# ----------------------------
# Handle genomes without any AMR genes
# ----------------------------
# For genomes with no AMR genes (all 0), Simpson distance is undefined.
# Here, we replace distances involving such genomes with 0.
amr_zero_genomes <- rowSums(amr_binary) == 0
amr_simpson_matrix[amr_zero_genomes, ] <- 0
amr_simpson_matrix[, amr_zero_genomes] <- 0

# ----------------------------
# Export distance matrices to CSV files
# ----------------------------
# Output files are saved under "output" directory using relative paths.
write.csv(amr_jaccard_matrix,  file = file.path(OUTPUT_DIR, "amr_jaccard_distance_matrix_all_genomes.csv"))
write.csv(amr_simpson_matrix, file = file.path(OUTPUT_DIR, "amr_simpson_distance_matrix_all_genomes.csv"))
write.csv(vf_jaccard_matrix,  file = file.path(OUTPUT_DIR, "vf_jaccard_distance_matrix_all_genomes.csv"))
write.csv(vf_simpson_matrix, file = file.path(OUTPUT_DIR, "vf_simpson_distance_matrix_all_genomes.csv"))
