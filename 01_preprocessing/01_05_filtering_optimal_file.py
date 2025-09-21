from pathlib import Path
import os
import pandas as pd
import re

# ---------------------------
# Configuration
# ---------------------------
ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR  = ROOT / "output" / "01_preparation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METADATA_CSV       = OUTPUT_DIR / "ecoli_genomes_25875_metadata.csv"
PHYLOGROUP_CSV     = OUTPUT_DIR / "ecoli_genomes_25875_phylogroup_ezclermont.csv"
ASSEMBLY_STATS_CSV = OUTPUT_DIR / "ecoli_genomes_25875_assembly_stats.csv"


# ----------------------------
# Load processed data
# ----------------------------
metadata_df = pd.read_csv(METADATA_CSV, index_col=0)
phylogroup_df = pd.read_csv(PHYLOGROUP_CSV, index_col=0)
assembly_stats_df = pd.read_csv(ASSEMBLY_STATS_CSV, index_col=0)

# ----------------------------------------------
# Filters: A–G, size 4.4–6.0 Mb, N50 ≥ 40 kb
# ----------------------------------------------
valid_phylogroups = ["A", "B1", "B2", "C", "D", "E", "F", "G"]

phy_ok  = phylogroup_df[phylogroup_df["Phylogroup"].isin(valid_phylogroups)]
asm_ok  = assembly_stats_df[
    (assembly_stats_df["Size"] >= 4.4e6) &
    (assembly_stats_df["Size"] <  6.0e6) &
    (assembly_stats_df["N50"]  >= 40000)
]

# Intersection of filenames passing all criteria
selected = sorted(set(phy_ok.index) & set(asm_ok.index))

# ----------------------------
# Save selected filenames
# ----------------------------
pd.DataFrame(selected, columns=["File"]).to_csv(OUTPUT_DIR / "ecoli_genomes_filtered_25080.csv", index=False)
