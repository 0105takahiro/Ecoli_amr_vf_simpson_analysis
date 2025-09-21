from pathlib import Path
import shutil
import subprocess

# ------------------------------------------------------------
# Set directory paths
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]
SNP_SITES_OUT_DIR = ROOT / "output" / "03_distance"/ "snp_distance" / "snp_sites_output"
OUTPUT_DIR        = ROOT / "output" / "03_distance"/ "snp_distance" / "snp_dists_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PHYLOGROUPS = ["A", "B1", "B2", "C", "D", "E", "F", "G"]

# ------------------------------------------------------------
# Run snp-dists for each phylogroup
# ------------------------------------------------------------
def run_snp_dists_for_phylogroup(phylogroup_name: str) -> None:
    """
    Run snp-dists on the multiple alignment for a phylogroup and write CSV.
    """
    aln = SNP_SITES_OUT_DIR / f"snp_sites_output_{phylogroup_name.lower()}.aln"
    if not aln.exists():
        raise FileNotFoundError(f"Alignment not found: {aln}")

    out_csv = OUTPUT_DIR / f"snp_dists_{phylogroup_name.lower()}.csv"

    # Ensure snp-dists is available
    if shutil.which("snp-dists") is None:
        raise RuntimeError("snp-dists not found in PATH. Activate the right conda env.")

    # snp-dists -c writes CSV to stdout → redirect to file
    cmd = ["snp-dists", "-c", str(aln)]
    with open(out_csv, "w") as fh:
        result = subprocess.run(cmd, stdout=fh, stderr=subprocess.PIPE, text=True, check=False)

    if result.returncode != 0:
        raise RuntimeError(f"snp-dists failed for {phylogroup_name}: {result.stderr}")

    print(f"Saved: {out_csv}")

if __name__ == "__main__":
    for pg in PHYLOGROUPS:
        run_snp_dists_for_phylogroup(pg)
