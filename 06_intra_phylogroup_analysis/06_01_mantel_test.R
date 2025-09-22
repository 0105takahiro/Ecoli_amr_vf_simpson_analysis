# ---------------------------
# Dependencies
# ---------------------------
suppressPackageStartupMessages({
  library(vegan)   # mantel
  library(dplyr)
  library(readr)
})

get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- "--file="
  path <- sub(file_arg, "", args[grep(file_arg, args)])
  if (length(path) == 1) return(dirname(normalizePath(path)))
}

SCRIPT_DIR <- get_script_dir()
ROOT <- normalizePath(file.path(SCRIPT_DIR, "..", ".."))

# ----------------------------
# Directories (override by CLI args if needed)
# ----------------------------
GENOTYPE_MATRIX_DIR <- file.path(ROOT, "output", "03_distance", "genotype_distance" , "genotypic_distance_matrix_each_phylogroup")
SNP_MATRIX_DIR      <- file.path(ROOT, "output", "03_distance", "snp_distance", "snp_dists_output")

OUTPUT_DIR_BASE     <- file.path(ROOT, "output", "06_intra_phylogroup_analysis")
OUTPUT_DIR          <- file.path(OUTPUT_DIR_BASE, "mantel_result")
dir.create(OUTPUT_DIR_BASE, recursive = TRUE, showWarnings = FALSE)
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

PERMUTATIONS        <- 1000L

# CLI: Rscript mantel_vegan.R [GENOTYPE_DIR] [SNP_DIR] [OUTPUT_DIR] [PERM]
args <- commandArgs(trailingOnly = TRUE)
if (length(args) >= 1) GENOTYPE_MATRIX_DIR <- args[1]
if (length(args) >= 2) SNP_MATRIX_DIR      <- args[2]
if (length(args) >= 3) OUTPUT_DIR          <- args[3]
if (length(args) >= 4) PERMUTATIONS        <- as.integer(args[4])

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
