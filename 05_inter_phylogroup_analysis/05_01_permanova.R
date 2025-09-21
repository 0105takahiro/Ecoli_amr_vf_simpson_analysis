# ---- Package (only vegan). Auto-install if missing ----
if (!requireNamespace("vegan", quietly = TRUE)) {
  install.packages("vegan", repos = "https://cloud.r-project.org")
}
suppressPackageStartupMessages({
  library(vegan)  # adonis2
})

get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- "--file="
  path <- sub(file_arg, "", args[grep(file_arg, args)])
  if (length(path) == 1) return(dirname(normalizePath(path)))
}

SCRIPT_DIR <- get_script_dir()
ROOT <- normalizePath(file.path(SCRIPT_DIR, "..", ".."))

DIST_DIR        <- file.path(ROOT, "output", "distance_matrix_all_phylogroups")
PREPARATION_DIR <- file.path(ROOT, "output", "preparation")
PHYLOGROUP_CSV  <- file.path(PREPARATION_DIR, "ecoli_genomes_filtered_25080_phylogroup.csv")


PERM <- 1000L
SEED <- 0L

FILES <- list(
  "AMR / Jaccard" = "amr_jaccard_distance_matrix_all_phylogroups.csv",
  "AMR / Simpson" = "amr_simpson_distance_matrix_all_phylogroups.csv",
  "VF  / Jaccard" = "vf_jaccard_distance_matrix_all_phylogroups.csv",
  "VF  / Simpson" = "vf_jaccard_distance_matrix_all_phylogroups.csv"
)

# ---- Helpers (base R only) ----
read_phylogroup_base <- function(path) {
  df <- utils::read.csv(path, header = TRUE, check.names = FALSE, stringsAsFactors = FALSE)
  rownames(df) <- df[[1]]
  df[[1]] <- NULL
  if (!"Phylogroup" %in% colnames(df)) {
    stop("Column 'Phylogroup' not found in: ", path)
  }
  df[["Phylogroup"]] <- as.factor(df[["Phylogroup"]])
  df
}

read_dist_sqcsv_base <- function(path) {
  # Read a square matrix CSV (first column = row names, header = column names) -> numeric matrix -> as.dist
  dt <- utils::read.csv(path, header = TRUE, check.names = FALSE, stringsAsFactors = FALSE)
  rn <- dt[[1]]
  dt[[1]] <- NULL

  # Coerce to numeric (suppress warnings if any non-numeric strings appear)
  for (j in seq_len(ncol(dt))) {
    dt[[j]] <- suppressWarnings(as.numeric(dt[[j]]))
  }
  m <- as.matrix(dt)
  rownames(m) <- rn
  colnames(m) <- colnames(dt)

  # Align on common row/column names (and unify order)
  common <- intersect(rownames(m), colnames(m))
  if (length(common) < 2L) stop("Distance matrix too small or row/col names don't match: ", path)
  m <- m[common, common, drop = FALSE]

  # Ensure diagonal zeros
  diag(m) <- 0

  # If any NA remains, try to drop rows/cols containing NA
  if (anyNA(m)) {
    warning("NA found in distance matrix; attempting to drop rows/cols with NA")
    keep <- (rowSums(is.na(m)) == 0) & (colSums(is.na(m)) == 0)
    if (sum(keep) < 2L) stop("Too many NAs in distance matrix: ", path)
    m <- m[keep, keep, drop = FALSE]
  }

  stats::as.dist(m)
}

# ---- Main ----
phy <- read_phylogroup_base(PHYLOGROUP_CSV)

cat("[PERMANOVA] GROUP=Phylogroup ",
    sprintf("PERM=%d SEED=%d\n", PERM, SEED), sep = "")

for (label in names(FILES)) {
  fp <- file.path(DIST_DIR, FILES[[label]])
  if (!file.exists(fp)) {
    cat(sprintf("  %-14s : FILE NOT FOUND -> %s\n", label, fp))
    next
  }

  # Read distance matrix (base R)
  d <- read_dist_sqcsv_base(fp)

  # Match order by labels of the dist object
  labs <- labels(d)
  missing_in_phy <- setdiff(labs, rownames(phy))
  if (length(missing_in_phy) > 0L) {
    cat(sprintf("  %-14s : WARNING %d samples not in phy table (ignored in model)\n",
                label, length(missing_in_phy)))
  }
  grp <- phy[labs, "Phylogroup", drop = TRUE]

  # Run PERMANOVA
  set.seed(SEED)
  fit <- try(vegan::adonis2(d ~ grp, permutations = PERM), silent = TRUE)

  if (inherits(fit, "try-error")) {
    cat(sprintf("  %-14s : ERROR (adonis2 failed)\n", label))
    rm(d, fit, grp, labs); gc()
  } else {
    tab <- as.data.frame(fit)
    eff <- tab[1, , drop = FALSE]
    R2  <- as.numeric(eff$R2)
    Fv  <- as.numeric(eff$F)
    pv  <- as.numeric(eff$`Pr(>F)`)
    n   <- length(labs)
    cat(sprintf("  %-14s : R2 = %.4f, F = %.3f, p = %.3g, n = %d\n",
                label, R2, Fv, pv, n))
    rm(d, fit, tab, eff, grp, labs); gc()
  }
}

cat("[DONE]\n")

