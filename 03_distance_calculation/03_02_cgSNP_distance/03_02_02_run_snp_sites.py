from pathlib import Path
import subprocess

# -----------------------------------------------------------
# Set directory paths
# -----------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]
ROARY_OUT_DIR     = ROOT / "output" / "03_distance"/ "snp_distance"/ "roary_output"
SNP_SITES_OUT_DIR = ROOT / "output" / "03_distance"/ "snp_distance" / "snp_sites_output"
SNP_SITES_OUT_DIR.mkdir(parents=True, exist_ok=True)

PHYLOGROUPS = ["A", "B1", "B2", "C", "D", "E", "F", "G"]

# ------------------------------------------------------------
# Run snp-sites for each phylogroup
# ------------------------------------------------------------
def run_snp_sites_for_phylogroup(phylogroup_name: str) -> None:
    """
    Run snp-sites on the Roary core gene alignment of a phylogroup.
    Input  : Output from Roary (core_gene_alignment.aln).
    Output : SNP alignment.
    """

    PROJECT_DIR = ROARY_OUT_DIR / f"roary_output_{phylogroup_name.lower()}"
    in_aln  = PROJECT_DIR / "core_gene_alignment.aln"
    out_aln = (SNP_SITES_OUT_DIR / f"snp_sites_output_{phylogroup_name.lower()}.aln").resolve()
    err_log = SNP_SITES_OUT_DIR / f"snp_sites_{phylogroup_name.lower()}.stderr.txt"

    # Run snp-sites
    proc = subprocess.run(
        ["snp-sites", "-o", str(out_aln), "core_gene_alignment.aln"],
        text=True,
        cwd=PROJECT_DIR,
        stdout=subprocess.PIPE
    )

for pg in PHYLOGROUPS:
    run_snp_sites_for_phylogroup(pg)