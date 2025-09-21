import pandas as pd
import numpy as np
import os
import re
import glob
import io
import sys
from pathlib import Path

# ----------------------------
# Configuration
# ----------------------------
# Define project root relative to the script location
ROOT = Path(__file__).resolve().parents[3]

# Input: Phylogroup assignment table
INPUT_DIR_1  = ROOT / "output" / "01_preparation"
PHYLOGROUP_CSV = INPUT_DIR_1 / "ecoli_genomes_filtered_25080_phylogroup.csv"
phylo = pd.read_csv(PHYLOGROUP_CSV, index_col=0)

# Input: distance matrices across all phylogroups
INPUT_DIR_2  = ROOT / "output" / "03_distance" / "genotype_distance" / "genotypic_distance_matrix_all_phylogroups"

# Output: distance matrices split by each phylogroup
OUTPUT_DIR = ROOT / "output" / "03_distance" / "genotype_distance" / "genotypic_distance_matrix_each_phylogroup"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Input distance matrices (pre-computed across all phylogroups)
AMR_JACCARD_CSV = INPUT_DIR_2 / "amr_jaccard_distance_matrix_all_phylogroups.csv"
AMR_SIMPSON_CSV = INPUT_DIR_2 / "amr_simpson_distance_matrix_all_phylogroups.csv"
VF_JACCARD_CSV  = INPUT_DIR_2 / "vf_jaccard_distance_matrix_all_phylogroups.csv"
VF_SIMPSON_CSV  = INPUT_DIR_2 / "vf_simpson_distance_matrix_all_phylogroups.csv"

# Load matrices
amr_jaccard = pd.read_csv(AMR_JACCARD_CSV, index_col=0)
amr_simpson = pd.read_csv(AMR_SIMPSON_CSV, index_col=0)
vf_jaccard  = pd.read_csv(VF_JACCARD_CSV, index_col=0)
vf_simpson  = pd.read_csv(VF_SIMPSON_CSV, index_col=0)


def export_phylogroup_distance_matrices(
    matrix_df: pd.DataFrame,          # square distance matrix (rows/columns = fasta names)
    pg_df: pd.DataFrame,              # index = fasta name, must contain column 'Phylogroup'
    gene: str = "amr",                # "amr" or "vf"
    metric: str = "jaccard",          # "jaccard" or "simpson"
    output_dir: str = "./distance_matrix_output",
    groups = ("A","B1","B2","C","D","E","F","G"),
):
    """
    Example:
        export_phylogroup_distance_matrices(amr_jaccard, phylogroup_df,
                                            gene="amr", metric="jaccard",
                                            output_dir="./out")

    Output files:
        ./out/amr_jaccard_distance_a.csv
        ./out/amr_jaccard_distance_b1.csv
        etc.
    """

    # Ensure string-type keys and restrict matrix to common members
    dist = matrix_df.copy()
    dist.index = dist.index.astype(str)
    dist.columns = dist.columns.astype(str)
    names = sorted(set(dist.index).intersection(dist.columns))
    dist = dist.loc[names, names]  # enforce square and consistent ordering

    pg = pg_df.copy()
    pg.index = pg.index.astype(str)

    # Generate sub-matrices and export per phylogroup
    for g in groups:
        members = [m for m in pg.index[pg["Phylogroup"] == g] if m in names]
        if len(members) == 0:
            print(f"[WARN] {g}: no members found in the distance matrix. Skipping.")
            continue
        members = sorted(members)
        sub = dist.loc[members, members]
        out_path = os.path.join(output_dir, f"{gene.lower()}_{metric.lower()}_distance_{g.lower()}_2.csv")
        sub.to_csv(out_path)
        print(f"[OK] {g}: {sub.shape[0]}×{sub.shape[1]} -> {out_path}")

# Iterate over AMR/VF distance matrices with both Jaccard and Simpson metrics
for data, gene, metric in zip(
    [amr_jaccard, amr_simpson, vf_jaccard, vf_simpson],
    ["amr", "amr", "vf", "vf"],
    ["jaccard", "simpson", "jaccard", "simpson"]
):
    export_phylogroup_distance_matrices(
        data,
        phylo,
        gene=gene,
        metric=metric,
        output_dir=OUTPUT_DIR
    )
