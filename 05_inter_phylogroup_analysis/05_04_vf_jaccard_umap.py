import pandas as pd
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

PHYLOGROUP_CSV = ROOT / "output" / "preparation" /"ecoli_genomes_filtered_25080_phylogroup.csv"
VF_CSV = ROOT / "output" / "amr_and_vf_genes_ok" / "vf_genes_presence_absence.csv"
VF_DISTANCE_MATRIX_CSV = ROOT / "output" / "distance_matrix_all_phylogroups" / "vf_jaccard_distance_matrix_all_phylogroups.csv"

phylogroup_df = pd.read_csv(PHYLOGROUP_CSV, index_col=0)
vf_df = pd.read_csv(VF_CSV, index_col=0)
vf_distance_matrix =pd.read_csv(VF_DISTANCE_MATRIX_CSV,index_col=0)

# UMAP dimensionality reduction
umap_model = umap.UMAP(
    random_state=0, n_neighbors=20, metric="precomputed",
    n_components=2, min_dist=0.25
)
embedding = umap_model.fit_transform(vf_distance_matrix)
embedding_df = pd.DataFrame(embedding, index=vf_distance_matrix.index, columns=["UMAP1", "UMAP2"])
embedding_merged = embedding_df.join(phylogroup_df).dropna()

# ----------------------------
# Scatter plot (no axes, no legend)
# ----------------------------
color_palette = ["#4670ef", "#f2c475", "#6ce187", "#f7605a",
                 "#a367f1", "#b98859", "#f89be5", "#bab8b8"]
custom_palette = sns.color_palette(color_palette)

# Generate UMAP plot with phylogroup coloring
plt.figure(figsize=(3.45,2.5))

ax=sns.scatterplot(
    x=embedding_merged["UMAP1"],y=embedding_merged["UMAP2"],
    s=0.8,
    hue=embedding_merged["Phylogroup"],
    hue_order=["A","B1","B2","C","D","E","F","G"],
    palette=custom_palette,
    linewidth=0.005,
    alpha=0.8)

# Calculate axis limits
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
xlength = xmax - xmin
ylength = ymax - ymin

ax.set_xlim(xmin-xlength*0.005, xmax+xlength*0.005)
ax.set_ylim(ymin-ylength*0.005, ymax+ylength*0.005)

ax.set_axis_off() 
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_frame_on(False)
ax.patch.set_edgecolor('none')

# Remove axes and pads
ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)

ax.get_legend().remove()

plt.savefig(
    FIGURE_OUTPUT_DIR / "umap_vf_jaccard.pdf" ,
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.0
    )
