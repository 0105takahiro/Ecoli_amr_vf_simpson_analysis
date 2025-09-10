import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from pathlib import Path

plt.rcParams['font.family'] = 'Arial'

# ----------------------------
# Configuration
# ----------------------------
ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = ROOT / "Output" / "gene_gene_distance"
FIGURE_OUTPUT_DIR = ROOT / "Figures"
FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FIGURE = FIGURE_OUTPUT_DIR / "genotype_distance_kde_plot.pdf"

AMR_JACCARD_CSV = INPUT_DIR / "gene_gene_distance_amr_jaccard.csv"
AMR_SIMPSON_CSV = INPUT_DIR / "gene_gene_distance_amr_simpson.csv"
VF_JACCARD_CSV  = INPUT_DIR / "gene_gene_distance_vf_jaccard.csv"
VF_SIMPSON_CSV  = INPUT_DIR / "gene_gene_distance_vf_simpson.csv"

amr_jaccard = pd.read_csv(AMR_JACCARD_CSV, index_col=0)
amr_simpson = pd.read_csv(AMR_SIMPSON_CSV, index_col=0)
vf_jaccard  = pd.read_csv(VF_JACCARD_CSV, index_col=0)
vf_simpson  = pd.read_csv(VF_SIMPSON_CSV, index_col=0)

phylogroup  = ['A','B1','B2','C','D','E','F','G']
palette_hex = ["#4670ef", "#f2c475", "#6ce187", "#f7605a", "#a367f1", "#b98859", "#f89be5", "#bab8b8"]
palette     = dict(zip(phylogroup, palette_hex))

fig, axes = plt.subplots(2, 2, figsize=(8.5, 5))

def _filter_valid_groups(df, xcol, group_col="Phylogroup"):
    """Helper to drop groups with n<2 or zero variance (not used below; keep for safety)."""
    d = df[[group_col, xcol]].dropna()
    g = d.groupby(group_col)[xcol].agg(["count", "std"])
    keep = g[(g["count"] >= 2) & (g["std"] > 0)].index
    return d[d[group_col].isin(keep)]

# ========= Top-left: Jaccard-AMR ==========
sns.kdeplot(
    data=amr_jaccard, x='Jaccard distance', hue="Phylogroup", palette=palette,
    linewidth=2.5, bw_adjust=2, cut=10, gridsize=500,
    common_norm=False, legend=False, ax=axes[0, 0], warn_singular=False
)
axes[0, 0].set_xlim(-0.05, 1.05)
axes[0, 0].set_ylim(0, 2.5)
axes[0, 0].set_yticks([])
axes[0, 0].set_yticklabels("")
axes[0, 0].set_ylabel("")
axes[0, 0].set_xticks([])
axes[0, 0].set_xticklabels("")
axes[0, 0].set_xlabel("")

# ========= Top-right: Simpson-AMR ==========
sns.kdeplot(
    data=amr_simpson, x='Simpson distance', hue="Phylogroup", palette=palette,
    linewidth=2.5, bw_adjust=2, cut=10, gridsize=500,
    common_norm=False, legend=False, ax=axes[0, 1], warn_singular=False
)
axes[0, 1].set_xlim(-0.05, 1.05)
axes[0, 1].set_ylim(0, 2.2)
axes[0, 1].set_yticks([])
axes[0, 1].set_yticklabels([])
axes[0, 1].set_ylabel("")
axes[0, 1].set_xticks([])
axes[0, 1].set_xticklabels("")
axes[0, 1].set_xlabel("")

# ========= Bottom-left: Jaccard-VF ==========
sns.kdeplot(
    data=vf_jaccard, x='Jaccard distance', hue="Phylogroup", palette=palette,
    linewidth=2.5, bw_adjust=2, cut=10, gridsize=500,
    common_norm=False, legend=False, ax=axes[1, 0], warn_singular=False
)
axes[1, 0].set_xlim(-0.05, 1.05)
axes[1, 0].set_ylim(0, 3.9)
axes[1, 0].set_yticks([])
axes[1, 0].set_yticklabels("")
axes[1, 0].set_ylabel("")
axes[1, 0].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
axes[1, 0].set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=9)
axes[1, 0].set_xlabel("")

# ========= Bottom-right: Simpson-VF ==========
sns.kdeplot(
    data=vf_simpson, x='Simpson distance', hue="Phylogroup", palette=palette,
    linewidth=2.5, bw_adjust=2, cut=10, gridsize=500,
    common_norm=False, legend=False, ax=axes[1, 1], warn_singular=False
)
axes[1, 1].set_xlim(-0.05, 1.05)
axes[1, 1].set_ylim(0, 42)
axes[1, 1].set_yticks([])
axes[1, 1].set_yticklabels([], fontsize=9)
axes[1, 1].set_ylabel("")
axes[1, 1].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
axes[1, 1].set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=9)
axes[1, 1].set_xlabel("")

# ========= Titles for each subpanel ==========
axes[0, 0].text(0.07, 1.1, "AMR Jaccard",  fontsize=10, ha='left', va='top', transform=axes[0, 0].transAxes)
axes[0, 1].text(0.07, 1.1, "AMR Simpson",  fontsize=10, ha='left', va='top', transform=axes[0, 1].transAxes)
axes[1, 0].text(0.07, 1.1, "VF Jaccard",   fontsize=10, ha='left', va='top', transform=axes[1, 0].transAxes)
axes[1, 1].text(0.07, 1.1, "VF Simpson",   fontsize=10, ha='left', va='top', transform=axes[1, 1].transAxes)

axes[0, 0].text(0.007, 1.11, "a", fontsize=12, fontweight='bold', ha='left', va='top', transform=axes[0, 0].transAxes)
axes[0, 1].text(0.007, 1.11, "b", fontsize=12, fontweight='bold', ha='left', va='top', transform=axes[0, 1].transAxes)
axes[1, 0].text(0.007, 1.11, "c", fontsize=12, fontweight='bold', ha='left', va='top', transform=axes[1, 0].transAxes)
axes[1, 1].text(0.007, 1.11, "d", fontsize=12, fontweight='bold', ha='left', va='top', transform=axes[1, 1].transAxes)

fig.text(0.4, -0.02, 'Genotypic distance', fontsize=15)
fig.text(-0.01, 0.55, 'Density', ha='center', va='top', fontsize=15, rotation=90)
fig.text(0.91, 0.87, 'Phylogroup', fontsize=15)

# ========= Legend ==========
legend_elements = [Line2D([0], [0], color=palette[label], lw=2.5, label=label) for label in phylogroup]
axes[0, 1].legend(
    handles=legend_elements, loc='center left', bbox_to_anchor=(1.01, 0.4),
    fontsize=9, frameon=False, ncol=1
)

plt.subplots_adjust(wspace=0.15, hspace=0.2)
plt.tight_layout()
plt.savefig(OUT_FIGURE, dpi=600, bbox_inches='tight')
