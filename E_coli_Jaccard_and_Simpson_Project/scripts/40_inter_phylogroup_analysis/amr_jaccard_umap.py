import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from pathlib import Path

# ----------------------------
# Configuration 
# ----------------------------
ROOT = Path(__file__).resolve().parents[2]

FIGURE_OUTPUT_DIR = ROOT/ "figures"/ "umap"
FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PHYLOGROUP_CSV = ROOT / "output" / "preparation" /"ecoli-genomes_filtered_25080_phylogroup.csv"
AMR_CSV = ROOT / "output" / "amr_and_vf_genes" / "amr-genes-presence-absence.csv"
AMR_DISTANCE_MATRIX_CSV = ROOT / "output" / "distance_matrix_all_phylogroups" / "amr_jaccard_distance_matrix_all_phylogroups.csv"

phylogroup_df = pd.read_csv(PHYLOGROUP_CSV, index_col=0)
amr_df = pd.read_csv(AMR_CSV, index_col=0)
amr_distance_matrix =pd.read_csv(AMR_DISTANCE_MATRIX_CSV,index_col=0)


# UMAP dimensionality reduction
umap_model = umap.UMAP(
    random_state=0, n_neighbors=20, metric='precomputed',
    n_components=2, min_dist=0.25
)
embedding = umap_model.fit_transform(amr_distance_matrix)
embedding_df = pd.DataFrame(embedding, index=amr_distance_matrix.index, columns=['UMAP1', 'UMAP2'])
embedding_merged = embedding_df.join(phylogroup_df).dropna()

# ----------------------------
# Scatter plot (no axes, no legend)
# ----------------------------
color_palette = ["#4670ef", "#f2c475", "#6ce187", "#f7605a",
                 "#a367f1", "#b98859", "#f89be5", "#bab8b8"]
custom_palette = sns.color_palette(color_palette)

# Generate UMAP plot with phylogroup coloring
plt.figure(figsize=(80, 60))
sns.set_context("paper")
sns.set_style("white")
ax = sns.scatterplot(
    x='UMAP1', y='UMAP2',
    data=embedding_merged,
    hue='Phylogroup',
    hue_order=['A','B1','B2','C','D','E','F','G'],
    palette=custom_palette,
    s=200,
    linewidth=0.1,
	alpha=0.8)

# Dynamically calculate axis limits
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
xlength = xmax - xmin
ylength = ymax - ymin

# Remove axes and pads
ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
ax.set_xlim(xmin-xlength*0.01, xmax+xlength*0.01)
ax.set_ylim(ymin-ylength*0.01, ymax+ylength*0.01)
ax.get_legend().remove()

plt.tight_layout()
plt.savefig(FIGURE_OUTPUT_DIR / "umap_projection_amr_jaccard.pdf", dpi=600)
plt.close()

# ----------------------------
# Legend-only figure
# ----------------------------
handles, labels = ax.get_legend_handles_labels()
fig_legend = plt.figure(figsize=(5, 10))
fig_legend.legend(handles,
                labels,
                loc='center', 
                frameon=False,
                fontsize=50, 
                ncol=1,
                handletextpad=0.8,
                handleheight=1.8, 
                borderaxespad=0.01,
				markerscale=5)  

# Remove axes from the legend-only figure
plt.axis('off')
plt.tight_layout()
plt.savefig(FIGURE_OUTPUT_DIR / "umap_legend.pdf", dpi=600, bbox_inches='tight')
plt.close()
