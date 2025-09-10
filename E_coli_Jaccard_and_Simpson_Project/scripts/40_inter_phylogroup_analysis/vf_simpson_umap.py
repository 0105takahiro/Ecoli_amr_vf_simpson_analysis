import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from pathlib import Path
import matplotlib as mpl
import numpy as np

# ============================================================
# Configuration 
# ============================================================
ROOT = Path(__file__).resolve().parents[2]

FIGURE_OUTPUT_DIR = ROOT/ "figures"/ "umap"
FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PHYLOGROUP_CSV = ROOT / "output" / "preparation" /"ecoli-genomes_filtered_25080_phylogroup.csv"
VF_CSV = ROOT / "output" / "amr_and_vf_genes" / "vf-genes-presence-absence.csv"
VF_DISTANCE_MATRIX_CSV = ROOT / "output" / "distance_matrix_all_phylogroups" / "vf_simpson_distance_matrix_all_phylogroups.csv"

# ============================================================
# Load input data
# ============================================================
phylogroup_df = pd.read_csv(PHYLOGROUP_CSV, index_col=0)
vf_df = pd.read_csv(VF_CSV, index_col=0)
vf_distance_matrix =pd.read_csv(VF_DISTANCE_MATRIX_CSV,index_col=0)

# ============================================================
# First UMAP embedding
# ============================================================
# Scatter plot (with manual cluster boundaries)
umap_model = umap.UMAP(
    random_state=0, n_neighbors=20, metric='precomputed',
    n_components=2, min_dist=0.25
)
embedding = umap_model.fit_transform(vf_distance_matrix)
embedding_df = pd.DataFrame(embedding, index=vf_distance_matrix.index, columns=['UMAP1', 'UMAP2'])
embedding_merged = embedding_df.join(phylogroup_df).dropna()


color_palette = ["#4670ef", "#f2c475", "#6ce187", "#f7605a",
                 "#a367f1", "#b98859", "#f89be5", "#bab8b8"]
custom_palette = sns.color_palette(color_palette)

# Generate UMAP plot with phylogroup coloring
plt.figure(figsize=(80, 60))
sns.set_context("paper")
sns.set_style("white")

ax = sns.scatterplot(
    data=embedding_merged,
    x='UMAP1', y='UMAP2',
    s=200,
    hue='Phylogroup',
    hue_order=['A','B1','B2','C','D','E','F','G'],
    palette=custom_palette,
    linewidth=0.1, alpha=0.8
)

# Ccalculate axis limits
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
xlength = xmax - xmin
ylength = ymax - ymin

# Draw cluster boundaries (these were manually determined based on UMAP projection)
ax.vlines(x=xmin+0.4, ymin=ymin+0.5, ymax=2.5, ls='--', lw=5, colors='gray')
ax.vlines(x=17, ymin=ymin+0.5, ymax=2.5, ls='--', lw=5, colors='gray')
ax.hlines(y=ymin+0.5, xmin=xmin+0.4, xmax=17, ls='--', lw=5, colors='gray')
ax.hlines(y=2.5, xmin=xmin+0.4, xmax=17, ls='--', lw=5, colors='gray')

ax.vlines(x=34, ymin=5, ymax=ymax-0.3, ls='--', lw=5, colors='gray')
ax.vlines(x=xmax-0.3, ymin=5, ymax=ymax-0.3, ls='--', lw=5, colors='gray')
ax.hlines(y=5, xmin=34, xmax=xmax-0.3, ls='--', lw=5, colors='gray')
ax.hlines(y=ymax-0.3, xmin=34, xmax=xmax-0.3, ls='--', lw=5, colors='gray')

# Clean axis
ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
ax.set_xlim(xmin-xlength*0.01, xmax+xlength*0.01)
ax.set_ylim(ymin-ylength*0.01, ymax+ylength*0.01)
ax.get_legend().remove()
plt.tight_layout()
plt.savefig(FIGURE_OUTPUT_DIR / "umap_projection_vf_simpson_1st.pdf", dpi=600, bbox_inches='tight')


# ============================================================
# Cluster assignment and composition
# ============================================================
cluster1 = embedding_merged[
    (embedding_merged['UMAP1'] >= xmin+0.4) & (embedding_merged['UMAP1'] <= 17) &
    (embedding_merged['UMAP2'] >= ymin+0.5) & (embedding_merged['UMAP2'] <= 2.5)
]
cluster2 = embedding_merged[
    (embedding_merged['UMAP1'] >= 34) & (embedding_merged['UMAP1'] <= xmax-0.3) &
    (embedding_merged['UMAP2'] >= 5) & (embedding_merged['UMAP2'] <= ymax-0.3)
]

cluster1 = cluster1.assign(cluster=1)
cluster2 = cluster2.assign(cluster=2)

clusters = pd.concat([cluster1, cluster2])

# Calculate phylogroup composition within each cluster
composition = pd.DataFrame()
for cluster_id in [1, 2]:
    subset = clusters[clusters['cluster'] == cluster_id]
    counts = subset['Phylogroup'].value_counts().reindex(['A','B1','B2','C','D','E','F','G']).fillna(0)
    composition = pd.concat([composition, counts.to_frame().T])
composition.index = [1, 2]


# ============================================================
# Pie chart plotting
# ============================================================
def pie_plot(cluster_num, cluster_name):
    subset = clusters[clusters['cluster'] == cluster_num]
    label = ['A','B1','B2','C','D','E','F','G']
    counts = subset['Phylogroup'].value_counts().reindex(label).fillna(0)
    OUT_PATH = os.path.join(FIGURE_OUTPUT_DIR, f"{cluster_name}_cluster_vf_simpson_pie_chart.pdf")
    plt.figure(figsize=(15, 15))
    plt.pie(counts, counterclock=False, startangle=90, labeldistance=None, colors=custom_palette)
    plt.savefig(OUT_PATH, dpi=600, bbox_inches='tight')

# Generate pie charts for both clusters
pie_plot(1, 'non_b2')
pie_plot(2, 'b2')


# ============================================================
# Gene prevalence analysis
# ============================================================
vf1 = vf_df.loc[cluster1.index]
vf2 = vf_df.loc[cluster2.index]

# Calculate prevalence of esp genes in each cluster
vf1_positive = vf1[(vf1['espL1']==1)|(vf1['espX4']==1)|(vf1['espX5']==1)]
vf2_negative = vf2[(vf2['espL1']==0)|(vf2['espX4']==0)|(vf2['espX5']==0)]

print("Cluster 1: % esp-positive:", np.round(len(vf1_positive) / len(vf1) * 100, 1))
print("Cluster 2: % esp-negative:", np.round(len(vf2_negative) / len(vf2) * 100, 1))


# ============================================================
# Second UMAP on non-B2 cluster
# ============================================================
vf_distance_matrix1 = vf_distance_matrix.loc[cluster1.index, cluster1.index]

mapper2 = umap.UMAP(random_state=0, n_neighbors=50, metric='precomputed',
                    n_components=2, min_dist=0.1)
embedding2 = mapper2.fit_transform(vf_distance_matrix1)

embedding_df2 = pd.DataFrame(embedding2, index=cluster1.index, columns=['UMAP1','UMAP2'])
embedding_merged2 = pd.merge(embedding_df2, phylogroup_df, how='inner', left_index=True, right_index=True).dropna()


# Scatter plot for cluster 1 (second embedding)
plt.rcParams['font.family'] = 'Arial'

plt.figure(figsize=(80, 60))
ax = sns.scatterplot(
    data=embedding_merged2,
    x='UMAP1', y='UMAP2',
    s=300, hue='Phylogroup',
    hue_order=['A','B1','B2','C','D','E','F','G'],
    palette=custom_palette,
    linewidth=0.1, alpha=0.8
)

# Calculate axis limits
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
plt.tight_layout()

# Draw thick black border around plot
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(7)

# Create legend 
legend = ax.legend(
    loc='upper left',
    bbox_to_anchor=(1.005, 0.92),
    prop={'size': 120},
    title='Phylogroup',
    title_fontsize=150,
    labelspacing=0.7,
    handletextpad=1.0,
    frameon=False
)

# Increase legend marker size
for handle in legend.legend_handles:
    handle.set_markersize(130) 

# Save figure
plt.savefig(OUT_FIGURE_2 = FIGURE_OUTPUT_DIR / "umap_projection_vf_simpson_2nd.pdf", dpi=600, bbox_inches='tight')
