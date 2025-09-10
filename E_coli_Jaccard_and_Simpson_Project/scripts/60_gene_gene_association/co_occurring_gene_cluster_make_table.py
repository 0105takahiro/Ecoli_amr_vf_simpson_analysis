# ============================
# Imports
# ============================
from pathlib import Path
import pickle
from collections import defaultdict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Arial'

# ============================
# Paths / I/O configuration
# ============================
ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = ROOT / "Output" / "co_occurring_gene_clusters"
OUTPUT_DIR  = ROOT / "Figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load precomputed clique results
with open(INPUT_DIR / "amr_clusters.pkl", "rb") as f:
    amr_clusters = pickle.load(f)
with open(INPUT_DIR / "vf_clusters.pkl", "rb") as f:
    vf_clusters = pickle.load(f)

phylogroups = ['A', 'B1', 'B2', 'C', 'D', 'E', 'F', 'G']

# ============================
# (Optional) quick summary printout
# ============================
# Show which co-occurring gene clusters appear in which phylogroups
def print_clique_coverage(clusters_dict, phylogroups):
    group_dict = defaultdict(set)
    for pg in phylogroups:
        for genes in clusters_dict.get(pg, []):
            group_dict[tuple(sorted(genes))].add(pg)
    for genes, groups in sorted(group_dict.items(), key=lambda x: x[0]):
        print(f"{list(genes)}: {len(groups)} ({','.join(sorted(groups))})")

print_clique_coverage(amr_clusters, phylogroups)
print_clique_coverage(vf_clusters,  phylogroups)

# ============================
# Common gene clusters to display as rows in the heatmaps
# ============================
common_clusters_amr = [
    ["aac(6')-Ib-cr5", 'blaOXA-1', 'catB3'],
    ['aadA2', 'dfrA12'],
    ['aadA5', 'dfrA17'],
    ['blaNDM-5', 'ble'],
    ['qacEdelta1', 'sul1'],
    ['parE_I529L', 'ptsI_V25I'],
]

common_clusters_vf = [
    ['entA', 'entB', 'entC', 'entD', 'entE', 'entF', 'entS'],
    ['fepA', 'fepB', 'fepC', 'fepD', 'fepG'],
    ['hlyA', 'hlyB', 'hlyC', 'hlyD'],
    ['iroB', 'iroC', 'iroD', 'iroE', 'iroN'],
    ['irp1', 'irp2'],
    ['iucA', 'iucB', 'iucC', 'iucD'],
    ['kpsD', 'kpsM'],
    ['sepD', 'sepL', 'sepQ/escQ', 'sepZ/espZ'],
    ['stx1B', 'stxA'],
    ['stx2A', 'stx2B'],
    ['yagV/ecpE', 'yagW/ecpD', 'yagX/ecpC', 'yagY/ecpB', 'yagZ/ecpA'],
    ['ybtA', 'ybtE', 'ybtP', 'ybtQ', 'ybtS', 'ybtT', 'ybtU', 'ybtX'],
]

# Human-readable (plain) row labels for the DataFrame indices
amr_index_labels = [
    "aac(6')-Ib-cr5, blaOXA-1, catB3",
    "aadA2, dfrA12",
    "aadA5, dfrA17",
    "blaNDM-5, ble",
    "qacEdelta1, sul1",
    "parE_I529L, ptsI_V25I",
]
vf_index_labels = [
    "entABCDEFS",
    "fepABCDG",
    "hlyABCD",
    "iroBCDEN",
    "irp12",
    "iucABCD",
    "kpsDM",
    "sepD, sepL, sepQ/escQ, sepZ/espZ",
    "stx1B, stxA",
    "stx2A, stx2B",
    "yagVWXYZ/ecpEDCBA",
    "ybtAEPQSTUX",
]

# ============================
# Key fix: order-insensitive clique membership
# - Normalize each clique to a frozenset so that ['a','b'] and ['b','a'] match.
# - Build presence/absence matrices using these normalized sets.
# ============================
def normalize_clusters(clusters_dict, phylogroups):
    """
    Convert each list-like clique to a frozenset so membership checks are order-insensitive.
    Returns: dict[phylogroup] -> set[frozenset(genes)]
    """
    return {
        pg: {frozenset(clq) for clq in clusters_dict.get(pg, [])}
        for pg in phylogroups
    }

def build_presence_df(common_clusters, clusters_norm, phylogroups, index_labels=None):
    """
    Build a presence/absence matrix (rows = co-occurring gene cluster, cols = phylogroups).
    `clusters_norm` must be the output of `normalize_clusters`.
    """
    rows = []
    for clq in common_clusters:
        key = frozenset(clq)  # ignore order of genes in the clique
        row = [1 if key in clusters_norm[pg] else 0 for pg in phylogroups]
        rows.append(row)
    df = pd.DataFrame(rows, columns=phylogroups)
    if index_labels is not None:
        df.index = index_labels
    return df

amr_clusters_norm = normalize_clusters(amr_clusters, phylogroups)
vf_clusters_norm  = normalize_clusters(vf_clusters,  phylogroups)

amr_data = build_presence_df(common_clusters_amr, amr_clusters_norm, phylogroups, index_labels=amr_index_labels)
vf_data  = build_presence_df(common_clusters_vf,  vf_clusters_norm,  phylogroups, index_labels=vf_index_labels)

# ============================
# LaTeX-style labels for y-ticks
# ============================
amr_labels = [
    r"$\mathit{aac(6'\!)}\text{-}Ib\text{-}cr5$, $\mathit{bla}_{\mathrm{OXA\text{-}1}}$, $\mathit{catB3}$",
    r"$\mathit{aadA2}$, $\mathit{dfrA12}$",
    r"$\mathit{aadA5}$, $\mathit{dfrA17}$",
    r"$\mathit{bla}_{\mathrm{NDM\text{-}5}}$, $\mathit{ble}$",
    r"$\mathit{qacEdelta1}$, $\mathit{sul1}$",
    r"$\mathit{parE}$ I529L, $\mathit{ptsI}$ V25I",
]
vf_labels = [
    r"$\mathit{entABCDEFS}$",
    r"$\mathit{fepABCDG}$",
    r"$\mathit{hlyABCD}$",
    r"$\mathit{iroBCDEN}$",
    r"$\mathit{irp12}$",
    r"$\mathit{iucABCD}$",
    r"$\mathit{kpsDM}$",
    r"$\mathit{sepD}$, $\mathit{sepL}$, $\mathit{sepQ}\text{/}\mathit{escQ}$, $\mathit{sepZ}\text{/}\mathit{espZ}$",
    r"$\mathit{stx1B}$, $\mathit{stxA}$",
    r"$\mathit{stx2A}$, $\mathit{stx2B}$",
    r"$\mathit{yagVWXYZ}\text{/}\mathit{ecpEDCBA}$",
    r"$\mathit{ybtAEPQSTUX}$",
]

# "Representative gene"  on the right side
represent_amr = [
    r"$\mathit{bla}_{\mathrm{OXA\text{-}1}}$",
    r"$\mathit{aadA2}$",
    r"$\mathit{dfrA17}$",
    r"$\mathit{ble}$",
    r"$\mathit{sul1}$",
    r"$\mathit{ptsI}$ V25I",
]
represent_vf = [
    r"$\mathit{entC}$",
    r"$\mathit{fepG}$",
    r"$\mathit{hlyD}$",
    r"$\mathit{iroB}$",
    r"$\mathit{irp2}$",
    r"$\mathit{iucB}$",
    r"$\mathit{kpsD}$",
    r"$\mathit{sepQ/escQ}$",
    r"$\mathit{stx1B}$",
    r"$\mathit{stx2A}$",
    r"$\mathit{yagV/ecpE}$",
    r"$\mathit{ybtA}$",
]

# ============================
# Plot heatmaps
# ============================
n_cols = len(phylogroups)
rep_x  = n_cols + 0.3     # x-position (in heatmap data coords) to place representative labels
shade_x = n_cols + 0.1    # left edge of the shaded "Rep. gene" column
shade_w = 2.0             # width of the shaded area

fig_height = 0.72 * (len(amr_data) + len(vf_data))
fig, ax = plt.subplots(
    2, 1, sharex=False,
    figsize=(20, fig_height),
    gridspec_kw={'height_ratios': [len(amr_data), len(vf_data)]}
)
fig.subplots_adjust(hspace=0.4)

# Draw heatmaps
sns.heatmap(amr_data, cmap='Blues', ax=ax[0], linewidths=1, cbar=False)
sns.heatmap(vf_data,  cmap='Blues', ax=ax[1], linewidths=1, cbar=False)

# Apply pretty y-tick labels
ax[0].set_yticklabels(amr_labels, fontsize=25)
ax[1].set_yticklabels(vf_labels, fontsize=25)

# Ticks styling
for a in ax:
    a.tick_params(pad=13, axis='x')
    a.tick_params(axis='x', which='major', labelsize=25,
                  labelbottom=False, bottom=False, top=False, labeltop=True)
    a.tick_params(axis='y', which='major', labelsize=25,
                  labelbottom=False, bottom=False, top=False, labeltop=True)

# Right-side "representative gene" labels
for idx, rep in enumerate(represent_amr):
    ax[0].text(rep_x, idx + 0.5, rep, fontsize=25, ha='left', va='center',
               transform=ax[0].transData)
for idx, rep in enumerate(represent_vf):
    ax[1].text(rep_x, idx + 0.5, rep, fontsize=25, ha='left', va='center',
               transform=ax[1].transData)

# Shaded column behind the representative labels
from matplotlib.patches import Rectangle
for axes, data in zip(ax, [amr_data, vf_data]):
    n_rows = data.shape[0]
    axes.add_patch(Rectangle((shade_x, -0.05),
                             shade_w, n_rows + 0.1,
                             color='lightgrey', zorder=0, clip_on=False, alpha=0.3))

# Column header and left-side panel labels
ax[0].text(shade_x + 0.05, -0.6, "Representative gene", fontsize=25,
           ha='left', va='bottom', transform=ax[0].transData, fontweight='bold')
ax[1].text(shade_x + 0.05, -0.6, "Representative gene", fontsize=25,
           ha='left', va='bottom', transform=ax[1].transData, fontweight='bold')

ax[0].text(-0.8, -0.6, "AMR", fontsize=25, ha='center', va='bottom',
           transform=ax[0].transData, fontweight='bold')
ax[1].text(-0.8, -0.6, "VF",  fontsize=25, ha='center', va='bottom',
           transform=ax[1].transData, fontweight='bold')

fig.subplots_adjust(hspace=0.4, left=0.2, right=0.85)

# Save figure
plt.savefig(OUTPUT_DIR / "phylogroup_invariant_cooccurring_gene_clusters_and_representative_genes.pdf",
            dpi=600, bbox_inches='tight')
