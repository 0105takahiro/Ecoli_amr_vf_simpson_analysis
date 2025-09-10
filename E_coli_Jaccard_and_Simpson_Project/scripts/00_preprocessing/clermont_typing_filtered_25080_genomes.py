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
OUTPUT_DIR  = ROOT / "output" / "preparation"

FILTERED_DATA_CSV = OUTPUT_DIR / "ecoli_genomes_filtered_25080.csv" 

# ----------------------------
# Pre-run checks
# ----------------------------
if shutil.which("ezclermont") is None:
    sys.exit("ERROR: 'ezclermont' not found in PATH. Activate the right conda env.")

# ----------------------------
# Parser for ezclermont output
# ----------------------------
def parse_ezclermont_output(text: str):
    text = (text or "").strip()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "\t" in line:
            toks = line.split("\t")
            if len(toks) >= 2 and toks[1].strip():
                return toks[1].strip()
        toks = line.split()
        if toks:
            return toks[-1].strip()
    return None

# -------------------------
# Determine Clermont typing
# -------------------------
files = pd.read_csv(FILTERED_DATA_CSV, index_col=0)
fasta_filenames = files.index.astype(str).tolist()

rows    = []
missing = []

for fname in fasta_filenames:
    norm_name = str(fname).strip()
    fasta_path = GENOMES_DIR / norm_name

    if not fasta_path.exists():
        missing.append(norm_name)
        rows.append({"File": norm_name, "Phylogroup": "Unresolved"})
        continue
    
	# Run ezclermont
    proc = subprocess.run(
        ["ezclermont", norm_name],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=GENOMES_DIR
    )

    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    pg = parse_ezclermont_output(combined)

    if pg:
        rows.append({"File": norm_name, "Phylogroup": pg})
        if proc.returncode != 0:
            stderr_snip = (proc.stderr or "").strip().splitlines()
            stderr_snip = " ".join(stderr_snip[:3])[:500]
            print(f"[WARN] ezclermont rc={proc.returncode} but parsed: {norm_name} stderr: {stderr_snip}")
    else:
        rows.append({"File": norm_name, "Phylogroup": "Unresolved"})
        stderr_snip = (proc.stderr or "").strip().splitlines()
        stderr_snip = " ".join(stderr_snip[:3])[:500]
        print(f"[ERROR] ezclermont failed: {norm_name} (rc={proc.returncode}) stderr: {stderr_snip}")

# ----------------------------
# Write outputs
# ----------------------------
pg_df = pd.DataFrame(rows)
pg_df = pg_df.sort_values("File").set_index("File")
pg_df.to_csv(OUTPUT_DIR / "ecoli_genomes_filtered_25080_phylogroup.csv")
