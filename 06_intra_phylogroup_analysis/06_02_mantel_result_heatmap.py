# === Import required packages ===
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats
import os
from pathlib import Path

# ---------------------------
# Configuration 
# ---------------------------
ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = ROOT / "output" / "06_intra_phylogroup_analysis" / "mantel_result"

FIGURE_OUTPUT_DIR = ROOT / "figures"
FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIGURE = FIGURE_OUTPUT_DIR / "mantel_result_heatmap.pdf"
OUT_FIGURE_LEGEND = FIGURE_OUTPUT_DIR / "mantel_heatmap_legend.pdf"

AMR_JACCARD_MANTEL_CSV = INPUT_DIR / "mantel_result_amr_jaccard.csv"
AMR_SIMPSON_MANTEL_CSV = INPUT_DIR / "mantel_result_amr_simpson.csv"
VF_JACCARD_MANTEL_CSV = INPUT_DIR / "mantel_result_vf_jaccard.csv"
VF_SIMPSON_MANTEL_CSV = INPUT_DIR / "mantel_result_vf_simpson.csv"

# Load correlation coefficients and p-values (alternating columns: coefficient, p-value)
amr_jaccard_df = pd.read_csv(AMR_JACCARD_MANTEL_CSV, index_col=0)
amr_simpson_df = pd.read_csv(AMR_SIMPSON_MANTEL_CSV, index_col=0)
vf_jaccard_df = pd.read_csv(VF_JACCARD_MANTEL_CSV, index_col=0)
vf_simpson_df = pd.read_csv(VF_SIMPSON_MANTEL_CSV, index_col=0)

# Merge all data into a single dataframe
df = pd.concat([
    amr_jaccard_df,
    amr_simpson_df,
    vf_jaccard_df,
    vf_simpson_df
], axis=1)

df = df.iloc[:,[2,3,6,7,10,11,14,15]]

df.columns=["AMR Jaccard rho","AMR Jaccard p",
            "AMR Simpson rho","AMR Simpson p",
            "VF Jaccard rho" ,"VF Jaccard p",
            "VF Simpson rho" ,"VF Simpson p"]

# Extract correlation coefficients for AMR and VF
amr_rho, vf_rho = df.iloc[:, [0, 2]], df.iloc[:, [4, 6]]
amr_rho.columns, vf_rho.columns = ["AMR Jaccard", "AMR Simpson"], ["VF Jaccard", "VF Simpson"]

# Extract corresponding p-values for AMR and VF
amr_p, vf_p = df.iloc[:, [1, 3]], df.iloc[:, [5, 7]]
amr_p.columns, vf_p.columns = ["AMR Jaccard", "AMR Simpson"], ["VF Jaccard", "VF Simpson"]

# ============================================================
# Multiple testing correction (Benjamini–Hochberg FDR)
# ============================================================
# Flatten all p-values into a 1D array
p_df = pd.concat([amr_p, vf_p], axis=1)
p_array = np.array(p_df).flatten()

# Apply false discovery rate correction (Benjamini-Hochberg)
q_array = stats.false_discovery_control(p_array, method="bh")
q_array = q_array.reshape(8, 4)

# Convert corrected q-values back into dataframe format
q_df = pd.DataFrame(q_array, index=p_df.index, columns=p_df.columns)

# Replace q-values with significance annotation: "*" for significant (q<0.05), "ns" for non-significant
q_df = q_df.where(q_df >= 0.05, 0)
q_df = q_df.where(q_df < 0.05, 1)
q_df = q_df.replace({0: "*", 1: "ns"})

# Separate annotations for AMR and VF for heatmap labeling
amr_q_df = q_df.iloc[:, 0:2]
vf_q_df  = q_df.iloc[:, 2:4]

# ============================================================
# Set colormap and normalization
# ============================================================
mpl.rcParams["font.family"] = "Arial"  # Use Arial font for publication
cmap = mpl.cm.BrBG
norm = mpl.colors.Normalize(vmin=-1, vmax=1)

# ============================================================
# Generate heatmap panel (AMR & VF side by side)
# ============================================================
fig = plt.figure(figsize=(2.3, 2.9))  # Narrower width, slightly taller height
gs = gridspec.GridSpec(2, 2, height_ratios=[0.05, 1], hspace=0.05, wspace=0.08)

# --- AMR Heatmap ---
ax1 = plt.subplot(gs[1, 0])
sns.heatmap(amr_rho, cmap=cmap, annot=amr_q_df, fmt="", annot_kws={"fontsize":7},
            linewidths=0.5, cbar=False, vmin=-1.0, vmax=1.0, ax=ax1)
ax1.set_yticklabels(["A","B1","B2","C","D","E","F","G"], fontsize=10, rotation=0)
ax1.set_xticklabels(["JD","SD"], fontsize=9, rotation=0)
ax1.set_title("AMR", fontsize=10, pad=5)
ax1.tick_params(axis="both", which="both", length=0)

# --- VF Heatmap ---
ax2 = plt.subplot(gs[1, 1])
sns.heatmap(vf_rho, cmap=cmap, annot=vf_q_df, fmt="", annot_kws={"fontsize":7},
            linewidths=0.5, cbar=False, vmin=-1.0, vmax=1.0, ax=ax2, yticklabels=False)
ax2.set_xticklabels(["JD","SD"], fontsize=9, rotation=0)
ax2.set_title("VF", fontsize=10, pad=5)
ax2.tick_params(axis="both", which="both", length=0)

# Export combined heatmap panel (without colorbar)
plt.savefig(OUT_FIGURE, dpi=600,bbox_inches='tight')

# ============================================================
# Generate colorbar panel
# ============================================================
fig_cb, ax_cb = plt.subplots(figsize=(0.15, 1.1))

cb = mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap, norm=norm, orientation="vertical")
cb.set_ticks([-1, -0.5, 0, 0.5, 1])
cb.set_ticklabels([-1, -0.5, 0, 0.5, 1], fontsize=8)

# Add vertical label
ax_cb.text(-1, 1.2, 'Spearman’s ρ', fontsize=10)

# Make colorbar frame thinner
for spine in ax_cb.spines.values():
    spine.set_linewidth(0.2)  

plt.savefig(OUT_FIGURE_LEGEND, dpi=600, bbox_inches="tight")
