from pathlib import Path
import subprocess
import pandas as pd
import io
import sys
import shutil

# -------------------------
# Config
# -------------------------
ROOT = Path(__file__).resolve().parents[2]

GENOMES_DIR = ROOT / "genomes"
OUTPUT_DIR  = ROOT / "output" / "01_preparation"

METADATA_CSV = OUTPUT_DIR / "ecoli_genomes_25875_metadata.csv"

if shutil.which("assembly-stats") is None:
    sys.exit("ERROR: 'assembly-stats' not found in PATH. Activate the right conda env.")

if not METADATA_CSV.exists():
    sys.exit(f"ERROR: Metadata CSV not found: {METADATA_CSV}")

meta = pd.read_csv(METADATA_CSV, index_col=0)
fasta_filenames = meta.index.tolist()

rows, missing = [], []

for fname in fasta_filenames:
    fasta_path = GENOMES_DIR / fname
    if not fasta_path.exists():
        missing.append(fname)
        continue

    proc = subprocess.run(
        ["assembly-stats", "-t", fname],
        text=True, stdout=subprocess.PIPE, cwd=GENOMES_DIR
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        print(f"[ERROR] assembly-stats failed: {fname}")
        continue

    try:
        df = pd.read_csv(io.StringIO(proc.stdout), sep="\t")
        if "filename" in df.columns:
            df["filename"] = df["filename"].map(lambda s: Path(str(s)).name)
        else:
            df["filename"] = fname
        rows.append(df)
    except Exception as e:
        print(f"[ERROR] parse failed: {fname} ({e})")

if not rows:
    sys.exit("[ERROR] No successful outputs. Aborting.")

stats = pd.concat(rows, ignore_index=True)

cols_map = {
    "filename": "File",
    "total_length": "Size",
    "number": "Contig number",
    "mean_length": "Average length",
    "longest": "Largest",
    "N50": "N50",
    "N50n": "L50",
    "N90": "N90",
    "N90n": "L90",
}
use_cols = [c for c in cols_map if c in stats.columns]
stats = stats[use_cols].rename(columns={k: v for k, v in cols_map.items() if k in use_cols})

numeric_cols = [c for c in ["Size","Contig number","Average length","Largest","N50","L50","N90","L90"] if c in stats.columns]
stats[numeric_cols] = stats[numeric_cols].apply(pd.to_numeric, errors="coerce")

stats = stats.set_index("File")
stats.to_csv(OUTPUT_DIR / "ecoli_genomes_25875_assembly_stats.csv")