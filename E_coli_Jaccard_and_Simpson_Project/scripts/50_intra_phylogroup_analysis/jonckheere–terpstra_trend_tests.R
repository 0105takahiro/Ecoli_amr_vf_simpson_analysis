# ----------------------------
# Dependencies
# ----------------------------
suppressPackageStartupMessages({
  library(clinfun)   # jonckheere.test
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
# Fixed paths (no defaults/aliases)
# ----------------------------
GENOTYPIC_DISTANCE_DIR <- file.path(
  ROOT, "output", "genotypic_distance_in_each_snp_range"
)
AMR_FILE <- file.path(
  GENOTYPIC_DISTANCE_DIR, "genotypic_distance_amr_in_each_cg_snp_distance.csv"
)
VF_FILE  <- file.path(
  GENOTYPIC_DISTANCE_DIR, "genotypic_distance_vf_in_each_cg_snp_distance.csv"
)

# Optional: allow overriding input files via CLI args (AMR, VF)
args <- commandArgs(trailingOnly = TRUE)
if (length(args) >= 2) {
  AMR_FILE <- args[1]
  VF_FILE  <- args[2]
}

cat("[INFO] AMR CSV: ", AMR_FILE, "\n", sep = "")
cat("[INFO]  VF CSV: ", VF_FILE,  "\n", sep = "")

# ----------------------------
# Expected schema and levels (strict, no normalization)
# ----------------------------
RANGE_COL      <- "SNP range"                       # exact column name
RANGE_LEVELS   <- c("1–10%", "0.1–1%", "0–0.1%")      # en-dash, not hyphen
METRIC_LEVELS  <- c("Jaccard", "Simpson")

read_input_strict <- function(path) {
  df <- readr::read_csv(path, show_col_types = FALSE)

  required_cols <- c(RANGE_COL, "Genotype_dist", "metric", "Phylogroup")
  missing <- setdiff(required_cols, names(df))
  if (length(missing) > 0) {
    stop(sprintf("Missing columns in %s: %s", basename(path), paste(missing, collapse = ", ")))
  }

  # Enforce exact factor levels (no normalization or auto-correction)
  df[[RANGE_COL]] <- factor(df[[RANGE_COL]], levels = RANGE_LEVELS, ordered = TRUE)
  if (!all(levels(df[[RANGE_COL]]) == RANGE_LEVELS)) {
    stop(sprintf("Unexpected '%s' levels in %s", RANGE_COL, basename(path)))
  }

  df$metric <- factor(df$metric, levels = METRIC_LEVELS)
  if (!all(levels(df$metric) == METRIC_LEVELS)) {
    stop(sprintf("Unexpected 'metric' levels in %s", basename(path)))
  }

  df
}

# ----------------------------
# Jonckheere–Terpstra for each (metric × phylogroup)
# ----------------------------
run_jt <- function(df, domain_label) {
  combos <- df %>% distinct(metric, Phylogroup) %>% arrange(metric, Phylogroup)

  res <- lapply(seq_len(nrow(combos)), function(i) {
    met <- combos$metric[i]
    pg  <- combos$Phylogroup[i]

    sub <- df %>% filter(metric == met, Phylogroup == pg)

    # Group counts in the fixed order
    n_vec <- sapply(RANGE_LEVELS, function(lvl) {
      sum(sub[[RANGE_COL]] == lvl, na.rm = TRUE)
    })

    # Monotone decreasing alternative
    jt <- suppressWarnings(
      jonckheere.test(sub$Genotype_dist, sub[[RANGE_COL]], alternative = "decreasing")
    )

    tibble(
      Domain     = domain_label,
      Metric     = as.character(met),
      Phylogroup = as.character(pg),
      JT_stat    = unname(jt$statistic),
      Z          = unname(jt$Z),
      P_value    = unname(jt$p.value),
      N_total    = nrow(sub),
      N_1_10     = n_vec[1],
      N_0_1_1    = n_vec[2],
      N_0_0_1    = n_vec[3]
    )
  }) %>% bind_rows()

  res
}

# ----------------------------
# Execute
# ----------------------------
amr <- read_input_strict(AMR_FILE)
vf  <- read_input_strict(VF_FILE)

res_amr <- run_jt(amr, "AMR")
res_vf  <- run_jt(vf,  "VF")

res_all <- bind_rows(res_amr, res_vf) %>%
  mutate(P_adj_BH = p.adjust(P_value, method = "BH")) %>%
  arrange(Domain, Metric, Phylogroup)

# ----------------------------
# Print only (no files saved)
# ----------------------------
print(res_all, n = nrow(res_all))
