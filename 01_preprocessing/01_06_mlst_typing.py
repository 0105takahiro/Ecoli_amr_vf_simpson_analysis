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
OUT_CSV           = OUTPUT_DIR / "ecoli_genomes_filtered_25080_MLST.csv"

# --- Tool check ---
if shutil.which("mlst") is None:
    sys.exit("ERROR: 'mlst' not found in PATH. Activate the right conda env.")

# --- Load filenames ---
files = pd.read_csv(FILTERED_DATA_CSV, index_col=0)
fasta_filenames = files.index.tolist()

# --- Helpers ---

def parse_mlst_output(text: str):
    """Extract ST from mlst output (supports TSV and CSV). Return str or None."""
    text = (text or "").strip()
    if not text:
        return None
    first = text.splitlines()[0]
    # try CSV first
    try:
        row = next(csv.reader([first]))
        if len(row) >= 3 and row[2].strip():
            return row[2].strip()
    except Exception:
        pass
    # fallback TSV
    parts = first.split("\t")
    if len(parts) >= 3 and parts[2].strip():
        return parts[2].strip()
    return None

# --- Main ---
rows, missing = [], []

for fname in fasta_filenames:
    fasta_path = GENOMES_DIR / fname
    if not fasta_path.exists():
        missing.append(fname)
        continue

    # Run mlst
    proc = subprocess.run([
        "mlst", fname
    ], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=GENOMES_DIR)

    if proc.returncode != 0 or not (proc.stdout or proc.stderr):
        print(f"[ERROR] mlst failed: {fname} (rc={proc.returncode})")
        continue

    st = parse_mlst_output(proc.stdout or proc.stderr)
    if not st:
        print(f"[ERROR] mlst parse failed: {fname}")
        continue

    rows.append({"File": fname, "ST": st})

if not rows:
    sys.exit("[ERROR] No successful mlst outputs. Aborting.")

st_df = pd.DataFrame(rows).sort_values("File").set_index("File")
st_df.to_csv(OUTPUT_DIR / "ecoli_genomes_filtered_25080_MLST.csv")