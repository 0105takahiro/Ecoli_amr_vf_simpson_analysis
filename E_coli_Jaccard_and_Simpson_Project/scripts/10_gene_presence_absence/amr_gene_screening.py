from pathlib import Path
import subprocess
import pandas as pd
import io
import sys
import shutil

# ----------------------------
# Configuration 
# ----------------------------
ROOT = Path(__file__).resolve().parents[2]

GENOMES_DIR = ROOT / "genomes"
OUTPUT_DIR  = ROOT / "output" / "amr_and_vf_genes"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILTERED_DATA_CSV = OUTPUT_DIR / "ecoli_genomes_filtered_25080.csv"
# Output: binary presence/absence matrix (rows: files, columns: AMR gene names)
OUT_CSV = OUTPUT_DIR / "amr_genes_presence_absence.csv"

# Column name constants (unify casing everywhere)
FILE_COL = "File"
GENE_COL = "Gene symbol"
COV_COL  = "% Coverage of reference sequence"
IDN_COL  = "% Identity to reference sequence"

# ----------------------------
# Pre-run checks
# ----------------------------
if shutil.which("amrfinder") is None:
    sys.exit("ERROR: 'amrfinder' not found in PATH. Activate the right conda env.")
if not FILTERED_DATA_CSV.exists():
    sys.exit(f"ERROR: Filter list CSV not found: {FILTERED_DATA_CSV}")

# ----------------------------
# Load files
# ----------------------------
files = pd.read_csv(FILTERED_DATA_CSV, index_col=0)
fasta_filenames = [str(x).strip() for x in files.index.tolist()]

# ----------------------------
# Run AMRFinder and collect results (no shell)
# ----------------------------
all_results = []
missing = []
processed = []

for fname in fasta_filenames:
    fasta_path = GENOMES_DIR / fname
    if not fasta_path.exists():
        missing.append(fname)
        continue

    # amrfinder -n <nucleotide_fasta> -O Escherichia
    proc = subprocess.run(
        ["amrfinder", "-n", fname, "-O", "Escherichia"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=GENOMES_DIR
    )

    if proc.returncode != 0:
        err1 = (proc.stderr or "").strip().splitlines()
        err1 = " ".join(err1[:3])[:500]
        print(f"[ERROR] amrfinder failed: {fname} (rc={proc.returncode}) stderr: {err1}")
        continue

    # Parse stdout (TSV). If empty, keep an empty frame.
    df = pd.read_csv(io.StringIO(proc.stdout), sep="\t") if proc.stdout.strip() else pd.DataFrame()

    if not df.empty:
        df[FILE_COL] = fname
        all_results.append(df)
    else:
        # No hits: add a placeholder row with only File for completeness
        all_results.append(pd.DataFrame({FILE_COL: [fname]}))

    processed.append(fname)

if not processed:
    sys.exit("[ERROR] No genomes were successfully processed. Aborting.")

# ----------------------------
# Filter hits and build presence/absence matrix
# ----------------------------
raw = pd.concat(all_results, ignore_index=True, sort=False) if all_results else pd.DataFrame(columns=[FILE_COL, GENE_COL])

# If coverage/identity columns exist, apply 70/90 thresholds
if COV_COL in raw.columns and IDN_COL in raw.columns:
    amr_calls = raw[(raw[COV_COL] >= 70) & (raw[IDN_COL] >= 90)].copy()
else:
    print("[WARN] Coverage/Identity columns not found; skipping 70%/90% filtering.")
    amr_calls = raw.copy()

# Build sorted gene list (unique)
if GENE_COL in amr_calls.columns:
    gene_list = sorted(g for g in amr_calls[GENE_COL].dropna().astype(str).unique() if g)
else:
    gene_list = []

# Initialize matrix with processed files only
amr_matrix = pd.DataFrame(0, index=processed, columns=gene_list, dtype=int)

# Mark presence = 1 where a gene is detected for a given file
if gene_list and GENE_COL in amr_calls.columns and FILE_COL in amr_calls.columns:
    for fname in processed:
        detected = set(amr_calls.loc[amr_calls[FILE_COL] == fname, GENE_COL].astype(str))
        if detected:
            amr_matrix.loc[fname, amr_matrix.columns.isin(detected)] = 1

amr_matrix.index.name = FILE_COL

# ----------------------------
# Save output
# ----------------------------
amr_matrix.to_csv(OUTPUT_DIR / "amr_genes_presence_absence.csv")
print(f"[OK] Saved: {OUTPUT_DIR / "amr_genes_presence_absence.csv".resolve()} (files={len(amr_matrix)}, genes={len(amr_matrix.columns)})")

if missing:
    print(f"[WARN] Missing FASTA files ({len(missing)}): first few → {missing[:5]}")
