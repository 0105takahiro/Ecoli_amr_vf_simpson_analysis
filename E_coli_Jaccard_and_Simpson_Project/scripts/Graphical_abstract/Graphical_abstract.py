# %%
# === Import required packages ===
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
from scipy.spatial.distance import squareform
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from scipy.cluster.hierarchy import linkage
# Set consistent font for publication
plt.rcParams['font.family'] = 'Arial'

#%%
# === Input toy example data ===

# Presence/absence matrix of 8 genes across 4 genomes
data = pd.DataFrame([
    [1, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 1],
], columns=['a','b','c','d','e','f','g'])

data.index = ['Genome 1','Genome 2','Genome 3','Genome 4']
data
#%%
# === Compute pairwise Jaccard and Simpson distances ===
genome_names = data.index.tolist()
n = len(genome_names)

# Initialize distance matrices
jaccard_matrix = np.zeros((n, n))
simpson_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        v1 = data.iloc[i].values
        v2 = data.iloc[j].values
        intersection = np.logical_and(v1, v2).sum()
        union = np.logical_or(v1, v2).sum()
        min_sum = min(v1.sum(), v2.sum())

        jaccard = 1 - intersection / union if union > 0 else 0
        simpson = 1 - intersection / min_sum if min_sum > 0 else 0

        jaccard_matrix[i, j] = jaccard
        simpson_matrix[i, j] = simpson

# Combine Jaccard (lower triangle) and Simpson (upper triangle) into one matrix
combined_matrix = np.full((n, n), np.nan)
for i in range(n):
    for j in range(n):
        if i > j:
            combined_matrix[i, j] = jaccard_matrix[i, j]
        elif i < j:
            combined_matrix[i, j] = simpson_matrix[i, j]

# Convert to DataFrame for seaborn heatmap
df_combined = pd.DataFrame(combined_matrix, index=genome_names, columns=genome_names)

#%%
%cd /Users/sekinetakahiro/reserch/zettlr/Figure/Graphical_abstract
# === Plot final dual-color heatmap with vertically stacked colorbars ===

plt.figure(figsize=(7, 7))  # ~13cm × 13cm size for pptx insertion

# --- Jaccard lower triangle ---
ax = sns.heatmap(
    df_combined,
    mask=~np.tril(np.ones_like(combined_matrix, dtype=bool), k=-1),
    cmap="Blues",
    vmin= 0, vmax= 0.8,
    square=True, cbar=False,
    linewidths=0.6, linecolor='grey'
)

# --- Simpson upper triangle ---
sns.heatmap(
    df_combined,
    mask=~np.triu(np.ones_like(combined_matrix, dtype=bool), k=1),
    cmap="Greens",
    vmin= 0, vmax= 0.8,
    square=True, cbar=False,
    linewidths=0.6, linecolor='grey',
    ax=ax
)

# --- Adaptive annotations (自動文字色) ---
for i in range(n):
    for j in range(n):
        val = combined_matrix[i, j]
        if not np.isnan(val):
            # Jaccard (下三角) / Simpson (上三角) 判定
            if i > j:
                cmap = plt.get_cmap("Blues")
            elif i < j:
                cmap = plt.get_cmap("Greens")
            else:
                continue  # 対角線は空白

            # 正規化して文字色決定
            norm_val = (val - vmin) / (vmax - vmin)
            rgba = cmap(norm_val)
            brightness = rgba[0]*0.299 + rgba[1]*0.587 + rgba[2]*0.114
            text_color = 'black' if brightness > 0.5 else 'white'

            ax.text(j+0.5, i+0.5, f"{val:.2f}", ha='center', va='center',
                    fontsize=20, color=text_color,fontweight='bold')

# --- Axis formatting ---
ax.set_xticklabels(genome_names, fontsize=15, rotation=0,fontweight='bold')
ax.set_yticklabels(genome_names, fontsize=15, rotation=0,fontweight='bold')
ax.tick_params(length=0)
for spine in ax.spines.values():
    spine.set_visible(False)

# --- Outer border ---
ax.add_patch(
    patches.Rectangle((0, 0), n, n, fill=False, edgecolor='black', linewidth=0.8, zorder=10)
)

# --- Diagonal separator ---
ax.plot([0, n], [0, n], color='black', lw=1)

# --- Create vertically stacked colorbars on right ---
divider = make_axes_locatable(ax)

# --- Layout adjustment ---
plt.subplots_adjust(left=0.20, right=0.85, top=0.92, bottom=0.15)

# --- Export for pptx ---
plt.savefig("graphical_abstract_for_genome_heatmap.pdf", dpi=600, bbox_inches='tight')
plt.show()

# %%
gene_names = data.columns.tolist()
n = len(gene_names)

# --- Jaccard & Simpson 距離を gene-gene で計算 ---
jaccard_matrix = np.zeros((n, n))
simpson_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        v1 = data.iloc[:, i].values
        v2 = data.iloc[:, j].values
        intersection = np.logical_and(v1, v2).sum()
        union = np.logical_or(v1, v2).sum()
        min_sum = min(v1.sum(), v2.sum())

        jaccard = 1 - intersection / union if union > 0 else 0
        simpson = 1 - intersection / min_sum if min_sum > 0 else 0

        jaccard_matrix[i, j] = jaccard
        simpson_matrix[i, j] = simpson

# --- Combine into one matrix ---
combined_matrix = np.full((n, n), np.nan)
for i in range(n):
    for j in range(n):
        if i > j:
            combined_matrix[i, j] = jaccard_matrix[i, j]
        elif i < j:
            combined_matrix[i, j] = simpson_matrix[i, j]

df_combined = pd.DataFrame(combined_matrix, index=gene_names, columns=gene_names)

# === Plotting ===
plt.figure(figsize=(3.5, 3.5))
vmin, vmax = 0, 1

# Base heatmap for Jaccard (lower triangle)
ax = sns.heatmap(
    df_combined,
    mask=~np.tril(np.ones_like(combined_matrix, dtype=bool), k=-1),
    cmap="Blues", vmin=vmin, vmax=vmax,
    square=True, cbar=False,
    linewidths=0.3, linecolor='grey'
)

# Overlay Simpson (upper triangle)
sns.heatmap(
    df_combined,
    mask=~np.triu(np.ones_like(combined_matrix, dtype=bool), k=1),
    cmap="Greens", vmin=vmin, vmax=vmax,
    square=True, cbar=False,
    linewidths=0.3, linecolor='grey',
    ax=ax
)

# Adaptive text color
for i in range(n):
    for j in range(n):
        val = combined_matrix[i, j]
        if not np.isnan(val):
            cmap = plt.get_cmap("Blues") if i > j else plt.get_cmap("Greens")
            norm_val = (val - vmin) / (vmax - vmin)
            rgba = cmap(norm_val)
            brightness = rgba[0]*0.299 + rgba[1]*0.587 + rgba[2]*0.114
            text_color = 'black' if brightness > 0.5 else 'white'
            ax.text(j+0.5, i+0.5, f"{val:.2f}", ha='center', va='center', fontsize=9, color=text_color)

# Axis formatting
ax.set_xticklabels(gene_names, fontsize=6, rotation=0)
ax.set_yticklabels(gene_names, fontsize=6, rotation=0)
ax.tick_params(length=0)
for spine in ax.spines.values():
    spine.set_visible(False)

# Outer border
ax.add_patch(patches.Rectangle((0, 0), n, n, fill=False, edgecolor='black', linewidth=0.8, zorder=10))

# Diagonal separator
ax.plot([0, n], [0, n], color='black', lw=0.6)

# Layout
plt.subplots_adjust(left=0.20, right=0.85, top=0.92, bottom=0.15)

# Save
plt.savefig("graphical_abstract_for_gene_heatmap.pdf", dpi=600, bbox_inches='tight')
plt.show()
#%%
import matplotlib.colors as mcolors
base_cmap = plt.get_cmap('RdYlBu_r')
soft_rdyblu = mcolors.LinearSegmentedColormap.from_list(
    'soft_rdyblu',
    base_cmap(np.linspace(0.15, 0.85, 100))
)
#%%
from matplotlib.colors import LogNorm
MARKER_SIZE = 18000 
TICK_FONTSIZE = 40
FDR_SIZE=340

def make_amr_scatter(data, gene1, gene2, fig, ax):
    x = len(gene2)
    y = len(gene1)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(-0.5, y - 0.5)
    ax.set_ylim(-0.5, x - 0.5)

    ax.set_xticks(np.arange(0, y, 1))
    ax.set_yticks(np.arange(0, x, 1))
    ax.set_xticklabels(gene2, size=TICK_FONTSIZE, rotation=90)
    ax.set_yticklabels(gene1, size=TICK_FONTSIZE)

    ax.invert_yaxis()

    for label_obj in ax.get_xticklabels():
        label_obj.set_color('black')
        label_obj.set_fontname("Arial")

    for label_obj in ax.get_yticklabels():
        label_obj.set_color('black')
        label_obj.set_fontname("Arial")

    for cutoff, size_rate in zip([0.025, 0.05, 0.075, 0.1, 1], [1, 0.58, 0.23, 0.07, 0.015]):
        if cutoff == 0.025:
            d = data[data['Simpson distance'] <= cutoff]
        elif cutoff == 1:
            d = data[data['Simpson distance'] > 0.1]
        else:
            d = data[(data['Simpson distance'] <= cutoff) & (data['Simpson distance'] > cutoff - 0.025)]

        if not d.empty:
            ax.scatter(
                list(d['Gene 2']), list(d['Gene 1']),
                s=[size_rate * MARKER_SIZE] * len(d),
                linewidths=0, cmap=soft_rdyblu, alpha=0.7,
                marker="s", c=d['Odds ratio'], norm=LogNorm(vmin=0.01, vmax=100)
            )

    thick = data[data['Jaccard distance'] <= 0.2]
    thin = data[data['Jaccard distance'] > 0.2]

    if not thick.empty:
        ax.scatter(thick['Gene 2'], thick['Gene 1'], s=MARKER_SIZE,
                   alpha=1, marker='s', facecolors='none',
                   linewidths=6, edgecolors='black')
    if not thin.empty:
        ax.scatter(thin['Gene 2'], thin['Gene 1'], s=MARKER_SIZE,
                   alpha=1, marker='s', facecolors='none',
                   linewidths=0.6, edgecolors='grey')

    for el1, el2 in zip([1, 2], [FDR_SIZE, FDR_SIZE * 7]):
        d = data[data['q-category'] == el1]
        ax.scatter(list(d['Gene 2']), list(d['Gene 1']),
                   s=[el2] * len(d), marker="+", c='Black')

    return fig, ax

# %%
df = pd.DataFrame([
    [4, 3, 0.07, 0.47, 10.0, 2],
    [2, 0, 0.00, 0.66, 1.03, 0],
    [2, 2, 0.39, 0.61, 3.81, 2],
    [2, 1, 0.09, 0.31, 15.85, 2],
    [3, 3, 0.06, 0.45, 14.5, 2],
    [4, 0, 0.01, 0.68, 0.23, 0],
    [4, 2, 0.38, 0.63, 3.72, 0],
    [4, 1, 0.29, 0.55, 6.46, 0],
    [4, 4, 0.42, 0.60, 5.30, 2],
    [1, 0, 0.01, 0.60, 0.33, 0],
    [0, 0, 0.00, 0.30, 50.0, 0],
    [3, 0, 0.00, 0.67, 0.99, 0],
    [1, 1, 0.02, 0.10, 100, 2],
    [3, 2, 0.03, 0.15, 20.0, 2],
    [3, 1, 0.24, 0.49, 10.15, 1]
], columns=[
    'Gene 1', 'Gene 2', 'Simpson distance', 'Jaccard distance',
    'Odds ratio', 'q-category'
])
df
#%%
dictionary={'Gene 1': {'gene 2': 0,
   'gene 3': 1,
   'gene 4': 2,
   'gene 5': 3,
   'gene 6': 4},
  'Gene 2': {'gene 1': 0,
   'gene 2': 1,
   'gene 3': 2,
   'gene 4': 3,
   'gene 5': 4},
  'index_to_gene1': {0: 'gene 2',
   1: 'gene 3',
   2: 'gene 4',
   3: 'gene 5',
   4: 'gene 6'},
  'index_to_gene2': {0: 'gene 1',
   1: 'gene 2',
   2: 'gene 3',
   3: 'gene 4',
   4: 'gene 5'}}
dictionary
#%%
height_ratios = [len(dictionary['Gene 1'])]
max_gene2_count=5
#%%
# サブプロット作成
%cd /Users/sekinetakahiro/reserch/scripts/Graphical_abstract
gene1_list=['gene 2','gene 3','gene 4','gene 5','gene 6']
gene2_list=['gene 1','gene 2','gene 3','gene 4','gene 5']

fig, ax = plt.subplots(1, 1, figsize=(12, 12))
make_amr_scatter(df, gene1_list, gene2_list, fig, ax)
fig.subplots_adjust(left=0.05, right=0.95)
plt.tight_layout()
plt.savefig("heatmap_for_graphical_abstrac.pdf", dpi=600, bbox_inches='tight')
plt.show()

# %%
# --- GA用：簡略化ヒートマップを合成して保存するスクリプト ---
# 依存: numpy, pandas, matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

# ----------------------------
# 設定
# ----------------------------
np.random.seed(42)  # 再現性
groups = ["A","B1","B2","C"]
# 各phylogroupの表示サンプル数（GAの視認性重視で均等に）
n_per_group={'A': 46, 'B1': 36, 'B2': 24, 'C': 30}
#%%
# 出力先
outdir = Path("./GA_heatmaps")
outdir.mkdir(parents=True, exist_ok=True)

# 色レンジ（0=同一/近い, 1=遠い）を固定して2図で比較しやすく
VMIN, VMAX = 0.0, 1.0

# ----------------------------
# ラベル（行列の並び）を作る
# ----------------------------
def make_labels(groups: List[str], n_per_group: Dict[str,int]) -> List[str]:
    labels = []
    for g in groups:
        labels += [f"{g}_{i:02d}" for i in range(1, n_per_group[g]+1)]
    return labels

labels = make_labels(groups, n_per_group)

# グループ境界（ブロック線用）
edges = np.cumsum([n_per_group[g] for g in groups])  # 12,24,36,...
edges = edges[:-1]  # 最後の端は不要

# ----------------------------
# 距離行列を合成（Jaccard的／Simpson的）
# ----------------------------
def synth_jaccard_like(groups, n_per_group, labels) -> pd.DataFrame:
    """
    Jaccard
    """
    N = sum(n_per_group.values())
    M = np.zeros((N, N), dtype=float)
    # グループごとに2〜3サブクラスターを作る
    start = 0
    for g in groups:
        n = n_per_group[g]
        end = start + n
        # サブクラスター割り当て
        n_sub = 8
        subs = np.random.randint(0, n_sub, size=n)
        # 同一グループ内の距離
        for i in range(n):
            for j in range(i+1, n):
                if subs[i] == subs[j]:
                    d = np.random.uniform(0.05, 0.2)
                else:
                    d = np.random.uniform(0.5, 0.9)
                M[start+i, start+j] = d
                M[start+j, start+i] = d
        start = end
    # 異なるグループ間の距離を大きめに
    start_i = 0
    for gi in groups:
        end_i = start_i + n_per_group[gi]
        start_j = 0
        for gj in groups:
            end_j = start_j + n_per_group[gj]
            if gi != gj:
                block = np.random.uniform(0.70, 1.0, size=(end_i-start_i, end_j-start_j))
                M[start_i:end_i, start_j:end_j] = block
            start_j = end_j
        start_i = end_i
    np.fill_diagonal(M, 0.0)
    return pd.DataFrame(M, index=labels, columns=labels)

def synth_simpson_like(groups, n_per_group, labels) -> pd.DataFrame:
    """
    Simpsonっぽく
    """
    N = sum(n_per_group.values())
    M = np.zeros((N, N), dtype=float)
    # グループごとに2〜3サブクラスターを作る
    start = 0
    for g in groups:
        n = n_per_group[g]
        end = start + n
        # サブクラスター割り当て
        n_sub = 3
        subs = np.random.randint(0, n_sub, size=n)
        # 同一グループ内の距離
        for i in range(n):
            for j in range(i+1, n):
                if subs[i] == subs[j]:
                    d = np.random.uniform(0.03, 0.1)
                else:
                    d = np.random.uniform(0.25, 0.45)
                M[start+i, start+j] = d
                M[start+j, start+i] = d
        start = end
    # 異なるグループ間の距離を大きめに
    start_i = 0
    for gi in groups:
        end_i = start_i + n_per_group[gi]
        start_j = 0
        for gj in groups:
            end_j = start_j + n_per_group[gj]
            if gi != gj:
                block = np.random.uniform(0.65, 1.0, size=(end_i-start_i, end_j-start_j))
                M[start_i:end_i, start_j:end_j] = block
            start_j = end_j
        start_i = end_i
    np.fill_diagonal(M, 0.0)
    return pd.DataFrame(M, index=labels, columns=labels)


D_jac = synth_jaccard_like(groups, n_per_group, labels)
D_sim = synth_simpson_like(groups, n_per_group, labels)
#%%
from scipy.cluster.hierarchy import linkage, leaves_list

def plot_reordered_heatmap(D, title: str, outfile_prefix: str, method: str = "average",
                           cmap="YlGnBu_r", show_colorbar=False):
    """
    D: 正方距離行列 (pd.DataFrame) 対称・対角0を想定
    """
    # 1) 距離→リンケージ（型はfloatに）
    arr = np.asarray(D.values, dtype=float)
    condensed = squareform(arr, checks=False)
    Z = linkage(condensed, method=method, optimal_ordering=True)

    # 2) 葉順で並べ替え（行・列同順）
    order = leaves_list(Z)
    D_ord = D.iloc[order, :].iloc[:, order]

    # 3) 樹形図なしのヒートマップ
    fig, ax = plt.subplots(figsize=(7, 7))
    sns.heatmap(D_ord, ax=ax, cmap=cmap, vmin=VMIN, vmax=VMAX,
                xticklabels=False, yticklabels=False,
                square=True, cbar=show_colorbar)

    fig.savefig(outdir / f"{outfile_prefix}.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / f"{outfile_prefix}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved: {outdir/outfile_prefix}.png / .pdf")
#%%
# ----------------------------
# 出力（2図）
# ----------------------------
plot_reordered_heatmap(D_jac, "Jaccard distance2 (reordered, no dendrogram)", "GA_cm_no_dend_jac")
plot_reordered_heatmap(D_sim, "Simpson distance2 (reordered, no dendrogram)", "GA_cm_no_dend_sim")

# %%
