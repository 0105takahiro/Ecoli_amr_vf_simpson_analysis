from pathlib import Path
import subprocess
import pandas as pd
import shutil
import sys
import os

# ----------------------------
# Config
# ----------------------------
ROOT = Path(__file__).resolve().parents[2]

GENOMES_DIR = ROOT / "genomes"
OUTPUT_DIR  = ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PHYLOGROUP_CSV = OUTPUT_DIR / "ecoli-genomes_26789_phylogroup-ezclermont.csv"
PROKKA_ROOT = OUTPUT_DIR / "output_prokka"
PROKKA_ROOT.mkdir(parents=True, exist_ok=True)

phylogrouops = ["A", "B1", "B2", "C", "D", "E", "F", "G"]

# ----------------------------
# Checks
# ----------------------------
if shutil.which("prokka") is None:
    sys.exit("ERROR: 'prokka' not found in PATH. Activate the right conda env.")
if not PHYLOGROUP_CSV.exists():
    sys.exit(f"ERROR: Input CSV not found: {PHYLOGROUP_CSV}")
if not GENOMES_DIR.exists():
    sys.exit(f"ERROR: Genomes directory not found: {GENOMES_DIR}")

# ----------------------------
# Load
# ----------------------------
df = pd.read_csv(PHYLOGROUP_CSV, index_col=0)
if "Phylogroup" not in df.columns:
    raise KeyError("Expected column 'Phylogroup' in input CSV")
df.index = df.index.map(lambda s: str(s).strip())

def stem(f):  # *.fasta -> basename
    return Path(f).stem

# ----------------------------
# Run Prokka
# ----------------------------
def run_prokka_for_phylogroup(phylogroup_name: str) -> None:
    genomes = [f for f in df.index[df["Phylogroup"] == phylogroup_name].tolist()]
    out_dir = PROKKA_ROOT / f"prokka_output_{phylogroup_name.lower()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Running Prokka for {phylogroup_name} (n={len(genomes)}) → {out_dir}")

    tmp_parent = PROKKA_ROOT / "tmp" / {phylogroup_name.lower()}
    tmp_parent.mkdir(parents=True, exist_ok=True)

    for fname in genomes:
        fasta_path = (GENOMES_DIR / fname).resolve()
        if not fasta_path.exists():
            print(f"[WARN] FASTA not found: {fasta_path}")
            continue

        prefix  = stem(fasta_path.name)
        tmp_dir = (tmp_parent / prefix).resolve()

        cmd = [
            "prokka",
            "--outdir", str(tmp_dir),   # Let Prokka create the directory (do not pre-create)
            "--prefix",  prefix,
            "--locustag","J001",
            "--increment","5",
            "--genus",   "Escherichia",
            "--species", "coli",
            "--force",                    # Overwrite if an output directory already exists
            str(fasta_path),              # Absolute input path
        ]
        proc = subprocess.run(
            cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

        if proc.returncode != 0:
            print(f"[ERROR] prokka failed: {fasta_path.name} (rc={proc.returncode})")
            if proc.stderr:
                print("==== STDERR ====")
                print(proc.stderr)  # full stderr
            if proc.stdout:
                print("==== STDOUT ====")
                print(proc.stdout)  # full stdout
            print(f"TMP kept: {tmp_dir}")
            try:
                print("TMP contents:", [p.name for p in tmp_dir.iterdir()])
            except FileNotFoundError:
                print("TMP missing")
            continue

        # Move only the .gff (fallback to the first *.gff if the expected name is not found)
        gff = tmp_dir / f"{prefix}.gff"
        if not gff.exists():
            gffs = list(tmp_dir.glob("*.gff"))
            if gffs:
                gff = gffs[0]

        if gff.exists():
            shutil.move(str(gff), str(out_dir / gff.name))
            shutil.rmtree(tmp_dir, ignore_errors=True)  # Clean up only on success
        else:
            print(f"[WARN] GFF not found for {fasta_path.name}")
            try:
                print("TMP contents:", [p.name for p in tmp_dir.iterdir()])
            except FileNotFoundError:
                print("TMP missing")

    print(f"[INFO] Completed {phylogroup_name}: output = {out_dir}")

for pg in phylogroups:
    run_prokka_for_phylogroup(pg)

print(f"[OK] Completed. Outputs under: {PROKKA_ROOT}")
