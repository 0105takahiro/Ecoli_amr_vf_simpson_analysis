from pathlib import Path
import subprocess
import pandas as pd
import io
import sys
import shutil
import csv

# ----------------------------
# Configuration
# ----------------------------
ROOT = Path(__file__).resolve().parents[2]
GENOMES_DIR = ROOT / "genomes"
OUTPUT_DIR  = ROOT / "output" / "01_preparation"

FILTERED_DATA_CSV = OUTPUT_DIR / "ecoli_genomes_filtered_25080.csv" 
PHYLOGROUP_DATA_CSV = OUTPUT_DIR / "ecoli_genomes_25875_phylogroup_ezclermont.csv"

filtered_file_df = pd.read_csv(FILTERED_DATA_CSV, index_col=0)
phylogroup_df = pd.read_csv(PHYLOGROUP_DATA_CSV, index_col=0)

# ----------------------------
# Write outputs
# ----------------------------
filtered_phylogroup = phylogroup_df[phylogroup_df.index.isin(filtered_file_df.index)]
filtered_phylogroup = filtered_phylogroup.sort_values('File')
filtered_phylogroup.to_csv(OUTPUT_DIR /"ecoli_genomes_filtered_25080_phylogroup_2.csv")
