from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import squareform
from matplotlib.patches import Patch
# Set consistent font for publication
plt.rcParams['font.family'] = 'Arial'

# ----------------------------
# Configuration 
# ----------------------------
ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = ROOT / "output"
GENOTYPIC_DISTANCE_DIR = INPUT_DIR / "genotypic_distance_matrix_each_phylogroup"
SNP_DISTANCE_DIR = INPUT_DIR / "snp_dists_output"

PHYLOGROUP_CSV = INPUT_DIR / "preparation" /"ecoli_genomes_filtered_25080_phylogroup.csv"
AMR_CSV = INPUT_DIR / "amr_and_vf_genes" / "amr_genes_presence_absence.csv"
VF_CSV = INPUT_DIR / "amr_and_vf_genes" / "vf_genes_presence_absence.csv"

OUTPUT_DIR = INPUT_DIR / "genotypic_distance_in_each_snp_range"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIGURE_OUTPUT_DIR = ROOT / "figures"
FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

phylogroup_df = pd.read_csv(PHYLOGROUP_CSV, index_col=0)
amr_df = pd.read_csv(AMR_CSV, index_col=0)
vf_df = pd.read_csv(VF_CSV, index_col=0)

# Split genomes by phylogroup
phylogroups = ['A','B1','B2','C','D','E','F','G']

amr_split = {pg: amr_df[amr_df.index.isin(phylogroup_df[phylogroup_df['Phylogroup']==pg].index)] for pg in phylogroups}
vf_split  = {pg: vf_df[vf_df.index.isin(phylogroup_df[phylogroup_df['Phylogroup']==pg].index)]  for pg in phylogroups}

# === Step 3: Load pairwise distance matrices ===
# AMR distances
distance_amr_jaccard = {
    pg: pd.read_csv(GENOTYPIC_DISTANCE_DIR / f"amr_jaccard_distance_{pg.lower()}.csv", index_col=0)
    for pg in phylogroups
}
distance_amr_simpson = {
    pg: pd.read_csv(GENOTYPIC_DISTANCE_DIR / f"amr_simpson_distance_{pg.lower()}.csv", index_col=0)
    for pg in phylogroups
}

# VF distances
distance_vf_jaccard = {
    pg: pd.read_csv(GENOTYPIC_DISTANCE_DIR / f"vf_jaccard_distance_{pg.lower()}.csv", index_col=0)
    for pg in phylogroups
}
distance_vf_simpson = {
    pg: pd.read_csv(GENOTYPIC_DISTANCE_DIR / f"vf_simpson_distance_{pg.lower()}.csv", index_col=0)
    for pg in phylogroups
}

# Convert distance matrices into condensed vectors
for pg in phylogroups:
    distance_amr_jaccard[pg] = squareform(distance_amr_jaccard[pg])
    distance_amr_simpson[pg] = squareform(distance_amr_simpson[pg])
    distance_vf_jaccard[pg]  = squareform(distance_vf_jaccard[pg])
    distance_vf_simpson[pg]  = squareform(distance_vf_simpson[pg])

# === Step 4: Load cgSNP distance matrices ===
snp_distances = {
    pg: pd.read_csv(SNP_DISTANCE_DIR / f"snp_dists_{pg.lower()}.csv", index_col=0)
    for pg in phylogroups
}

# Convert each dataframe to a square form matrix
for pg in phylogroups:
    snp_distances[pg] = squareform(snp_distances[pg])

# Compute cgSNP thresholds for each percentile (10%, 1%, 0.1%)
snp_thresholds = {
    pg: {
        '10%': np.percentile(snp_distances[pg], 10),
        '1%':  np.percentile(snp_distances[pg], 1),
        '0.1%': np.percentile(snp_distances[pg], 0.1)
    } 
    for pg in phylogroups
}

print('cgSNP thresholds for each cgSNP percentile')
for pg in phylogroups:
    print(pg,snp_thresholds[pg])

# === Step 5: Categorize pairwise distances by cgSNP bins ===

# Unified function to assign distance groups
def assign_snp_bin(genotype_dist, snp_dist, metric, phylogroup):
    th = snp_thresholds[phylogroup]
    bins = []
    for val_g, val_s in zip(genotype_dist, snp_dist):
        if val_s <= th['0.1%']:
            bins.append( ('0–0.1%', val_g) )
        elif val_s <= th['1%']:
            bins.append( ('0.1–1%', val_g) )
        elif val_s <= th['10%']:
            bins.append( ('1–10%', val_g) )
    df = pd.DataFrame(bins, columns=['SNP range', 'Genotype_dist'])
    df['metric'] = metric
    df['Phylogroup'] = phylogroup
    return df

# Generate full dataframe for AMR and VF
amr_data, vf_data = [], []
for pg in phylogroups:
    amr_data.append(assign_snp_bin(distance_amr_jaccard[pg], snp_distances[pg], 'Jaccard', pg))
    amr_data.append(assign_snp_bin(distance_amr_simpson[pg], snp_distances[pg], 'Simpson', pg))
    vf_data.append(assign_snp_bin(distance_vf_jaccard[pg],  snp_distances[pg], 'Jaccard', pg))
    vf_data.append(assign_snp_bin(distance_vf_simpson[pg],  snp_distances[pg], 'Simpson', pg))

combined_amr = pd.concat(amr_data, ignore_index=True)
combined_vf  = pd.concat(vf_data,  ignore_index=True)

# Ensure consistent ordering of group levels
group_order = ["1–10%", "0.1–1%", "0–0.1%"]
combined_amr['SNP range'] = pd.Categorical(combined_amr['SNP range'], categories=group_order, ordered=True)
combined_vf['SNP range']  = pd.Categorical(combined_vf['SNP range'], categories=group_order, ordered=True)

OUT_CSV_AMR = OUTPUT_DIR / "genotypic_distance_amr_in_each_cg_snp_distance.csv"
OUT_CSV_VF = OUTPUT_DIR / "genotypic_distance_vf_in_each_cg_snp_distance.csv"

combined_amr.to_csv(OUT_CSV_AMR)
combined_vf.to_csv(OUT_CSV_VF)

# === Step 6: Generate combined boxplots of AMR and VF genotype distances ===

OUT_FIGURE_BOXPLOT = FIGURE_OUTPUT_DIR / "boxplot_of_genotypic_distance_in_each_snp_bins.pdf"

# --- Common settings ---
metrics = ["Jaccard", "Simpson"]
snp_ranges = ['1–10%', '0.1–1%', '0–0.1%']
color_map = {'1–10%': 'blue', '0.1–1%': 'lightblue', '0–0.1%': 'green'}

# --- Generate cluster labels for sorting ---
tick_labels = []
for pg in phylogroups:
    for met in metrics:
        for snp_range in snp_ranges:
            tick_labels.append(f"{pg}_{met}_{snp_range}")

# --- Add cluster labels to both AMR and VF dataframes ---
for combined_df in [combined_amr, combined_vf]:
    combined_df['Cluster'] = combined_df['Phylogroup'].astype(str) + "_" + combined_df['metric'].astype(str) + "_" + combined_df['SNP range'].astype(str)
    combined_df['Cluster'] = pd.Categorical(combined_df['Cluster'], categories=tick_labels, ordered=True)
    combined_df.sort_values('Cluster', inplace=True)
    combined_df['Cluster'] = combined_df['Cluster'].astype(str)

# --- Color list for box coloring ---
box_colors = [color_map[label.split('_')[-1]] for label in tick_labels]

# --- Calculate boxplot positions ---
positions = []
pos = 0
for pg in phylogroups:
    for m_idx, m in enumerate(metrics):
        for s_idx, snp_range in enumerate(snp_ranges):
            positions.append(pos)
            if s_idx < 2:
                pos += 0.16
        if m_idx < 1:
            pos += 0.44
    pos += 0.75

# --- Create combined boxplot figure ---
fig, axs = plt.subplots(2, 1, figsize=(6.0, 2.7),
                        sharex=True,
                        gridspec_kw={'hspace': 0.05},
                        constrained_layout=True)

# --- Upper panel: AMR genotype distances ---
ax_amr = axs[0]
box_amr = ax_amr.boxplot(
    [combined_amr[combined_amr['Cluster'] == label]['Genotype_dist'] for label in tick_labels],
    positions=positions,
    patch_artist=True,
    widths=0.13,
    flierprops=dict(marker='o', markersize=0.2, linestyle='none',alpha=0.5),
    medianprops=dict(color='red', linewidth=1.0),
    boxprops=dict(linewidth=0.4),
    whiskerprops=dict(linewidth=0.4),
    capprops=dict(linewidth=0.4)
)

# --- Apply colors to AMR boxes ---
for patch, flier, color in zip(box_amr['boxes'], box_amr['fliers'], box_colors):
    patch.set_facecolor(color)
    flier.set_markerfacecolor(color)
    flier.set_markeredgecolor(color)

# --- Axis settings (AMR) ---
ax_amr.set_xlim(-0.35, positions[-1] + 0.35)
ax_amr.set_ylim(-0.02, 1.02)
ax_amr.set_ylabel("AMR distance", fontsize=10, labelpad=7)
ax_amr.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
ax_amr.set_yticklabels([0,0.2,0.4,0.6,0.8,1.0], fontsize=9)
ax_amr.spines['top'].set_visible(False)
ax_amr.spines['right'].set_visible(False)
ax_amr.set_xticks([])

# --- Add phylogroup labels above AMR panel ---
phylogroup_positions = []
for i in range(0, len(positions), 6):
    group_pos = np.mean(positions[i:i+6])
    phylogroup_positions.append(group_pos)
for pos_pg, label in zip(phylogroup_positions, phylogroups):
    ax_amr.text(pos_pg, 1.02, label, ha='center', va='bottom', fontsize=10, transform=ax_amr.get_xaxis_transform())

# --- Lower panel: VF genotype distances ---
ax_vf = axs[1]
box_vf = ax_vf.boxplot(
    [combined_vf[combined_vf['Cluster'] == label]['Genotype_dist'] for label in tick_labels],
    positions=positions,
    patch_artist=True,
    widths=0.13,
    flierprops=dict(marker='o', markersize=0.2, linestyle='none',alpha=0.5),
    medianprops=dict(color='red', linewidth=1.5),
    boxprops=dict(linewidth=0.4),
    whiskerprops=dict(linewidth=0.4),
    capprops=dict(linewidth=0.4)
)

# --- Apply colors to VF boxes ---
for patch, flier, color in zip(box_vf['boxes'], box_vf['fliers'], box_colors):
    patch.set_facecolor(color)
    flier.set_markerfacecolor(color)
    flier.set_markeredgecolor(color)

# --- Axis settings (VF) ---
ax_vf.set_xlim(-0.35, positions[-1] + 0.35)
ax_vf.set_ylim(-0.02, 1.02)
ax_vf.set_ylabel("VF distance", fontsize=10, labelpad=7)
ax_vf.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
ax_vf.set_yticklabels([0,0.2,0.4,0.6,0.8,1.0], fontsize=9)
ax_vf.spines['top'].set_visible(False)
ax_vf.spines['right'].set_visible(False)
ax_amr.set_xticks([])

# --- Add Jaccard/Simpson labels under VF panel ---
type_label_positions = []
type_label_texts = []
for i in range(0, len(positions), 6):
    jaccard_center = np.mean(positions[i:i+3])
    simpson_center = np.mean(positions[i+3:i+6])
    type_label_positions.extend([jaccard_center, simpson_center])
    type_label_texts.extend(["JD", "SD"])
for pos_type, text in zip(type_label_positions, type_label_texts):
    ax_vf.text(pos_type, -0.05, text, ha='center', va='top', fontsize=9)

# --- Legend (right side, combined into the same PDF) ---

legend_items = [
    ("1–10%",  color_map['1–10%']),
    ("0.1–1%", color_map['0.1–1%']),
    ("0–0.1%", color_map['0–0.1%']),
]
handles = [Patch(facecolor=c, edgecolor='black', linewidth=0.6) for _, c in legend_items]
labels  = [lab for lab, _ in legend_items]

# Place legend
legend = fig.legend(
    handles, labels,
    title="cgSNP distance",
    loc="center left",
    bbox_to_anchor=(0.89, 0.35),
    frameon=False,
    fontsize=8,
    title_fontsize=9,
    handlelength=1.6,
    handletextpad=0.3,
    labelspacing=0.3
)

plt.setp(legend.get_title(), fontname="Arial")
for t in legend.get_texts():
    t.set_fontname("Arial")

# --- Save figure (boxplot + legend together) ---
fig.savefig(OUT_FIGURE_BOXPLOT, dpi=600, bbox_inches='tight')
plt.close(fig)

# === Step 7: Build unified long-format dataframe ===
# Create flat-table converter function
def build_flat_dataframe(jaccard_dict, simpson_dict, snp_dict, gene_type):
    flat_list = []
    for pg in phylogroups:
        jaccard_array = jaccard_dict[pg]
        simpson_array = simpson_dict[pg]
        snp_array = snp_dict[pg]

        th = snp_thresholds[pg]

        # Classify cgSNP distance bins
        for val_j, val_s, val_snp in zip(jaccard_array, simpson_array, snp_array):
            if val_snp <= th['0.1%']:
                snp_range = '0–0.1%'
            elif val_snp <= th['1%']:
                snp_range = '0.1–1%'
            elif val_snp <= th['10%']:
                snp_range = '1–10%'
            else:
                continue  # outside threshold, ignore
            
            flat_list.append({
                'Phylogroup': pg,
                'SNP range': snp_range,
                'Jaccard': val_j,
                'Simpson': val_s,
                'Gene type': gene_type
            })

    return pd.DataFrame(flat_list)

# Apply for AMR and VF
flat_amr = build_flat_dataframe(distance_amr_jaccard, distance_amr_simpson, snp_distances, "AMR")
flat_vf  = build_flat_dataframe(distance_vf_jaccard, distance_vf_simpson, snp_distances, "VF")

# Merge
flat_all = pd.concat([flat_amr, flat_vf], ignore_index=True)

# === Function to calculate proportion of Jaccard=0 pairs among Simpson=0 pairs ===
def calculate_jaccard_zero_proportion(df):
    # Subset: only retain pairs where Simpson distance is zero
    filtered = df[df['Simpson'] == 0].copy()

    # For each combination of SNP group, phylogroup, and dataset (AMR/VF), calculate percentage of Jaccard=0
    summary = (
        filtered
        .groupby(['SNP range', 'Phylogroup', 'Gene type'])
        .apply(lambda x: (x['Jaccard'] == 0).mean() * 100)
        .reset_index(name='Proportion')
    )
    return summary

# Apply the function to entire flattened dataframe (which includes both AMR and VF)
summary_all = calculate_jaccard_zero_proportion(flat_all)

# Define the desired ordering of SNP bins for plotting (left-to-right order on x-axis)
group_order = ["1–10%", "0.1–1%", "0–0.1%"]
summary_all['SNP range'] = pd.Categorical(summary_all['SNP range'], categories=group_order, ordered=True)

color = ["#4670ef", "#f2c475e7", "#6ce187d6", "#f7605a","#a367f1ef", "#b988593a", "#f89be5", "#bab8b886"]
palette = sns.color_palette(color)

# === Step 8: Generate publication-quality lineplot from summary_all ===

OUT_FIGURE_LINEPLOT = FIGURE_OUTPUT_DIR / "proportion_of_identical_genotype_among_simpson_zero_pairs.pdf"

# Define panel dimensions for pptx export (fixed size)
total_width_inch = 2.9
height_inch = 2.7
panel_width_inch = total_width_inch / 2

# Create seaborn FacetGrid: separate panels for AMR and VF
g = sns.FacetGrid(
    summary_all,
    col='Gene type',
    sharey=True,
    height=height_inch,
    aspect=panel_width_inch / height_inch
)

# Draw line plots for each phylogroup
g.map_dataframe(
    sns.lineplot,
    x='SNP range',
    y='Proportion',
    hue='Phylogroup',
    marker='o',
    palette=palette,  # previously defined color palette
    linewidth=1.4,
    markersize=2.8
)

# Customize axes for each panel
for ax in g.axes.flat:
    ax.tick_params(axis='x', labelsize=8, length=3, width=0.8)
    ax.tick_params(axis='y', labelsize=8, length=3, width=0.8)
    ax.set_xlim(-0.2, 2.2)
    for label in ax.get_xticklabels():
        label.set_rotation(35)
        label.set_horizontalalignment('right')

# Set panel titles
g.axes[0][0].set_title("AMR", fontsize=10, pad=2.0)
g.axes[0][1].set_title("VF", fontsize=10, pad=2.0)

# Y-axis settings
g.set_ylabels("Proportion of Jaccard=0\namong Simpson=0 pairs (%)", fontsize=9)
g.axes[0][1].set_yticklabels([])
yticks = [0, 20, 40, 60, 80, 100]
g.axes[0][0].set_yticks(yticks)
g.axes[0][0].set_yticklabels([str(y) for y in yticks], fontsize=8)

# Remove X-axis labels (since group labels are plotted as tick labels)
g.set_xlabels("")

# Configure legend (shared across both panels)
legend = g.add_legend(
    title='Phylogroup',
    label_order=phylogroups,
    bbox_to_anchor=(0.77, 0.28), 
    loc='lower left',
    borderaxespad=0,
    fontsize=9,
    labelspacing=0.4,
)
g._legend.set_title("Phylogroup", prop={'size': 10})
for handle in g._legend.legend_handles:
    handle.set_linewidth(1.0)

# Add common X-axis label at bottom center
g.fig.text(0.5, 0.01, "cgSNP distance range", ha='center', va='center', fontsize=9)

plt.subplots_adjust(wspace=0.28)

# Export high-resolution figure (pptx optimized)
g.savefig( OUT_FIGURE_LINEPLOT, dpi=600, bbox_inches='tight')
