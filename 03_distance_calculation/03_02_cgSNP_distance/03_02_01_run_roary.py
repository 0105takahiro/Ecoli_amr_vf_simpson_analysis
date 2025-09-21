import subprocess
from pathlib import Path

# ------------------------------------------------------------
# Set directory paths and load genome file list
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]

OUTPUT_DIR = ROOT / "output" / "03_distance" / "snp_distance" 

PHYLOGROUPS = ["A", "B1", "B2", "C", "D", "E", "F", "G"]

# ------------------------------------------------------------
# Define function to run Roary for each phylogroup
# ------------------------------------------------------------
def run_roary_for_phylogroup(phylogroup_name):
    # Set output directory for this phylogroup
    PROJECT_DIR = (ROOT / "output" / "02_gene_screening" /"output_prokka"/ f"prokka_output_{phylogroup_name.lower()}").resolve()
    ROARY_OUTPUT_DIR = (ROOT / "output" / "03_distance"/ "snp_distance"/ "roary_output" / f"roary_output_{phylogroup_name.lower()}").resolve()

    gffs = sorted(p.name for p in PROJECT_DIR.glob("*.gff"))
    if not gffs:
        print(f"[WARN] no GFF in {PROJECT_DIR}")
        return

    cmd = [
        "roary", "-e", "--mafft", "-cd", "99", "-g", "500000000",
        "-f", str(ROARY_OUTPUT_DIR), "-p", "6", *gffs
    ]
    proc = subprocess.run(cmd, text=True,cwd=PROJECT_DIR)
    if proc.returncode != 0:
        print("[ERROR] roary failed:", proc.stderr)
    else:
        print(f"[OK] {phylogroup_name} -> {ROARY_OUTPUT_DIR}")

for pg in ["A", "B1", "B2", "C", "D", "E","F","G"]:
    run_roary_for_phylogroup(pg)
