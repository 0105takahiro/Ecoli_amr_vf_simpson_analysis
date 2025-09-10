# ----------------------------
# Dependencies
# ----------------------------
suppressPackageStartupMessages({
  library(vegan)   # mantel
  library(dplyr)
  library(readr)
})

# ----------------------------
# Project root resolution (folder name must be PROJECT_NAME)
# ----------------------------
PROJECT_NAME <- "E_coli_Jaccard_and_Simpson_Project"

get_script_path <- function() {
  # When executed via Rscript:
  args <- commandArgs(trailingOnly = FALSE)
  i <- grep("^--file=", args)
  if (length(i) > 0) return(normalizePath(sub("^--file=", "", args[i])))
  # When sourced interactively:
  if (!is.null(sys.frames()[[1]]$ofile)) {
    return(normalizePath(sys.frames()[[1]]$ofile))
  }
  stop("Unable to determine script path. Run via Rscript or source this file directly.")
}

resolve_project_root <- function(script_path) {
  p <- dirname(script_path)
  repeat {
    if (basename(p) == PROJECT_NAME) return(p)
    parent <- dirname(p)
    if (parent == p) break
    p <- parent
  }
  stop(sprintf("Project root '%s' not found above: %s", PROJECT_NAME, script_path))
}

SCRIPT_PATH <- get_script_path()
ROOT <- resolve_project_root(SCRIPT_PATH)
cat(sprintf("[INFO] PROJECT ROOT: %s\n", ROOT))

# ----------------------------
# Directories (override by CLI args if needed)
# ----------------------------
GENOTYPE_MATRIX_DIR <- file.path(ROOT, "output", "genotypic_distance_matrix_each_phylogroup")
SNP_MATRIX_DIR      <- file.path(ROOT, "output", "snp_dists_output")
OUTPUT_DIR          <- file.path(ROOT, "output", "mantel_result")
PERMUTATIONS        <- 1000L

# CLI: Rscript mantel_vegan.R [GENOTYPE_DIR] [SNP_DIR] [OUTPUT_DIR] [PERM]
args <- commandArgs(trailingOnly = TRUE)
if (length(args) >= 1) GENOTYPE_MATRIX_DIR <- args[1]
if (length(args) >= 2) SNP_MATRIX_DIR      <- args[2]
if (length(args) >= 3) OUTPUT_DIR          <- args[3]
if (length(args) >= 4) PERMUTATIONS        <- as.integer(args[4])
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

cat("[INFO] GENOTYPE DIR: ", GENOTYPE_MATRIX_DIR, "\n", sep = "")
cat("[INFO] SNP DIR     : ", SNP_MATRIX_DIR,      "\n", sep = "")
cat("[INFO] OUTPUT DIR  : ", OUTPUT_DIR,          "\n", sep = "")
cat("[INFO] PERMUTATIONS: ", PERMUTATIONS,        "\n", sep = "")

# ----------------------------
# File patterns
# ----------------------------
PHYLOGROUPS <- c("A","B1","B2","C","D","E","F","G")
METRICS     <- c("Jaccard","Simpson")
GENE_TYPES  <- c("AMR","VF")

# Genotype: "<gene_type>_<metric>_distance_<pg>.csv"
genotype_path <- function(metric, gene_type, pg) {
  gt <- tolower(switch(toupper(gene_type),
                       AMR = "amr",
                       VF  = "vf",
                       stop("gene_type must be 'AMR' or 'VF'")))
  file.path(
    GENOTYPE_MATRIX_DIR,
    sprintf("%s_%s_distance_%s.csv", gt, tolower(metric), tolower(pg))
  )
}

# SNP: "snp_dists_<pg>.csv"
snp_path <- function(pg) {
  file.path(SNP_MATRIX_DIR, sprintf("snp_dists_%s.csv", tolower(pg)))
}

# ----------------------------
# IO helpers
# ----------------------------
read_sqmat <- function(path) {
  if (!file.exists(path)) stop(sprintf("File not found: %s", path))
  m <- as.matrix(read.table(path, header = TRUE, row.names = 1, sep = ",", check.names = FALSE))
  if (!all(rownames(m) %in% colnames(m)) || !all(colnames(m) %in% rownames(m))) {
    stop(sprintf("Row/column names are inconsistent in %s", basename(path)))
  }
  m[rownames(m), rownames(m), drop = FALSE]
}

align_mats <- function(A, B) {
  ids <- intersect(rownames(A), rownames(B))
  if (length(ids) < 3) stop("Too few shared genomes between matrices.")
  list(A = A[ids, ids, drop = FALSE], B = B[ids, ids, drop = FALSE], n = length(ids))
}

run_mantel <- function(snp_mat, geno_mat, permutations) {
  vegan::mantel(as.dist(snp_mat), as.dist(geno_mat),
                method = "spearman", permutations = permutations, na.rm = TRUE)
}

# ----------------------------
# Output the Mantel results
# ----------------------------
write_results_csv <- function(df, path) {
  out <- df %>%
    dplyr::select(Phylogroup, GeneType, Metric, rho, p_value) %>%
    dplyr::rename(`Gene type` = GeneType, `p-value` = p_value)
  rownames(out) <- out$Phylogroup
  out$Phylogroup <- NULL
  utils::write.csv(out, file = path, row.names = TRUE)
}

# ----------------------------
# Mantel tests for all Gene types × Metrics × Phylogroups
# ----------------------------
for (gene_type in GENE_TYPES) {
  for (metric in METRICS) {
    cat(sprintf("\n[INFO] Running Mantel: %s / %s\n", gene_type, metric))
    acc <- vector("list", length(PHYLOGROUPS))

    for (i in seq_along(PHYLOGROUPS)) {
      pg <- PHYLOGROUPS[i]
      snp_mat <- read_sqmat(snp_path(pg))
      gen_mat <- read_sqmat(genotype_path(metric, gene_type, pg))

      aligned   <- align_mats(snp_mat, gen_mat)
      mantelres <- run_mantel(aligned$A, aligned$B, PERMUTATIONS)

      acc[[i]] <- tibble::tibble(
        Phylogroup = pg,
        GeneType   = gene_type,
        Metric     = metric,
        rho        = unname(mantelres$statistic),  # correlation
        p_value    = unname(mantelres$signif)      # p-value
      )

      cat(sprintf("  %-2s: rho = % .4f, p = %.3g, n = %d\n",
                  pg, acc[[i]]$rho, acc[[i]]$p_value, aligned$n))
    }

    df <- dplyr::bind_rows(acc)
    out_name <- sprintf("mantel_result_%s_%s.csv",
                        tolower(gene_type), tolower(metric))
    write_results_csv(df, file.path(OUTPUT_DIR, out_name))
  }
}

cat("\n[INFO] Done.\n")
