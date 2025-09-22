# =============================================================================
# Imports
# =============================================================================
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy import stats 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm

plt.rcParams['font.family'] = 'Arial'
# ----------------------------
# Configuration
# ----------------------------
ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR_1 = ROOT / "output" / "01_preparation"
INPUT_DIR_2 = ROOT / "output" / "02_gene_screening" / "amr_and_vf_genes"
INPUT_DIR_3 = ROOT / "output" / "07_gene_gene_association" / "gene_gene_distance"
INPUT_DIR_4 = ROOT / "output" / "07_gene_gene_association" / "co_occurring_gene_clusters" 


OUTPUT_DIR_1 = ROOT / "output" / "07_gene_gene_association" / "gene_gene_distance"
OUTPUT_DIR_1.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_2 = ROOT / "figures"
OUTPUT_DIR_2.mkdir(parents=True, exist_ok=True)

PHYLOGROUP_CSV = INPUT_DIR_1 / "ecoli_genomes_filtered_25080_phylogroup.csv"
AMR_CSV        = INPUT_DIR_2 / "amr_genes_presence_absence.csv"
VF_CSV         = INPUT_DIR_2 / "vf_genes_presence_absence.csv"
AMR_JACCARD_CSV = INPUT_DIR_3 / "gene_gene_distance_amr_jaccard.csv"
AMR_SIMPSON_CSV = INPUT_DIR_3 / "gene_gene_distance_amr_simpson.csv"
VF_JACCARD_CSV  = INPUT_DIR_3 / "gene_gene_distance_vf_jaccard.csv"
VF_SIMPSON_CSV  = INPUT_DIR_3 / "gene_gene_distance_vf_simpson.csv"

AMR_CLUSTER_PKL = INPUT_DIR_4 / "amr_clusters.pkl"
VF_CLUSTER_PKL  = INPUT_DIR_4 / "vf_clusters.pkl"
AMR_REPRESENT_PKL = INPUT_DIR_4 / "represent_gene_amr.pkl"
VF_REPRESENT_PKL  = INPUT_DIR_4 / "represent_gene_vf.pkl"

# =============================================================================
# Load Input Tables & Pickle Objects
# =============================================================================

phylogroup_df = pd.read_csv(PHYLOGROUP_CSV, index_col=0)

amr_df = pd.read_csv(AMR_CSV, index_col=0)   # presence/absence matrix (rows=genomes, cols=genes)
vf_df  = pd.read_csv(VF_CSV,  index_col=0)   # presence/absence matrix (rows=genomes, cols=genes)

amr_jaccard = pd.read_csv(AMR_JACCARD_CSV, index_col=0)
amr_simpson = pd.read_csv(AMR_SIMPSON_CSV, index_col=0)
vf_jaccard  = pd.read_csv(VF_JACCARD_CSV,  index_col=0)
vf_simpson  = pd.read_csv(VF_SIMPSON_CSV,  index_col=0)

with open(AMR_CLUSTER_PKL, "rb") as f:
    amr_clusters = pickle.load(f)
with open(VF_CLUSTER_PKL, "rb") as f:
    vf_clusters = pickle.load(f)

with open(AMR_REPRESENT_PKL, "rb") as f:
    amr_represent = pickle.load(f)
with open(VF_REPRESENT_PKL, "rb") as f:
    vf_represent = pickle.load(f)


# =============================================================================
# Domain Constants: Operons & Phylogroups
# =============================================================================
afa=['afaB-I','afaC-I']
ces=['cesAB','cesD','cesD2','cesF','cesL','cesT']
chu=['chuA','chuS','chuT','chuU','chuV','chuW','chuX','chuY']
ent=['entA','entB','entC','entD', 'entE', 'entF', 'entS']
esc=['escC', 'escD', 'escE', 'escF', 'escG', 'escI', 'escJ',
    'escL', 'escN', 'escO', 'escP', 'escR', 'escS', 'escT', 'escU', 'escV']
esp=['espA', 'espB', 'espD', 'espF', 'espG', 'espH', 'espJ', 'espK', 'espL1',
	'espL2', 'espL4', 'espM1', 'espM2', 'espN', 'espP', 'espR1', 'espR3',
    'espR4', 'espW', 'espX1', 'espX2', 'espX4', 'espX5', 'espX6', 'espX7/nleL',
    'espY1', 'espY2', 'espY3', 'espY4']
fep=['fepA', 'fepB', 'fepC', 'fepD', 'fepG']
fim=['fimA', 'fimB', 'fimC', 'fimD', 'fimE', 'fimF', 'fimG', 'fimH', 'fimI']
gsp=['gspC', 'gspD', 'gspE', 'gspF', 'gspG', 'gspH', 'gspI', 'gspJ', 'gspK', 'gspL', 'gspM']
hly=['hlyA', 'hlyB', 'hlyC', 'hlyD']
iro=['iroB', 'iroC', 'iroD', 'iroE', 'iroN']
irp=['irp1','irp2']
iuc=['iucA', 'iucB', 'iucC', 'iucD']
kps=['kpsD', 'kpsM', 'kpsT']
nle=['nleA', 'nleA/espI', 'nleB1', 'nleB2', 'nleC', 'nleD', 'nleE', 'nleF', "nleG7'", 'nleH1', 'nleH2']
pap=['papB', 'papC', 'papD', 'papE', 'papF', 'papG', 'papH', 'papI', 'papJ', 'papK', 'papX']
sep=['sepD', 'sepL', 'sepQ/escQ', 'sepZ/espZ']
sfa=['sfaB', 'sfaC', 'sfaD', 'sfaE', 'sfaF', 'sfaG', 'sfaX', 'sfaY']
shu=['shuA', 'shuS', 'shuT', 'shuX', 'shuY']
stx=['stx1A', 'stx1B', 'stx2A', 'stx2B', 'stxA']
yag=['yagV/ecpE', 'yagW/ecpD', 'yagX/ecpC', 'yagY/ecpB', 'yagZ/ecpA']
ybt=['ybtA', 'ybtE', 'ybtP', 'ybtQ', 'ybtS', 'ybtT', 'ybtU', 'ybtX']

operons=[afa,ces,chu,ent,esc,esp,fep,fim,gsp,hly,iro,irp,iuc,kps,nle,pap,sep,
		sfa,shu,stx,yag,ybt]

phylogroups=['A','B1','B2','C','D','E','F','G']


# =============================================================================
# Remove non-Representative Genes within Co-occurring Clusters
# =============================================================================

def remove_genes(data, gene_clusters, represent_genes):
    """
    Remove gene pairs that include non-representative genes from co-occurring  gene clusters.

    Parameters:
    - data: DataFrame containing gene-gene distance or association data.
            Must include 'Phylogroup', 'Gene 1', and 'Gene 2' columns.
    - gene_clusters: Dictionary mapping each phylogroup to a list of co-occurring gene clusters.
    - represent_genes: List of representative genes to retain.

    Returns:
    - all_df: Filtered DataFrame retaining only gene pairs where both genes
              are representative genes within their respective phylogroup.
    """
    all_df=pd.DataFrame()
    for phylogroup in phylogroups:
        # Subset the data for the phylogroup
        df_phylogroup=data[data['Phylogroup']==phylogroup]

        # Get all gene clusters for this phylogroup
        clusters_phylogroup=gene_clusters[phylogroup]

        non_need_genes = [
            gene
            for sublist in clusters_phylogroup 
            for gene in sublist
            if gene not in represent_genes
        ]
        # Remove gene pairs if either gene is in the non-representative list
        need_df_phylogroup = df_phylogroup[
            ~(df_phylogroup['Gene 1'].isin(non_need_genes) |
              df_phylogroup['Gene 2'].isin(non_need_genes))
        ]
        all_df=pd.concat([all_df,need_df_phylogroup])
    return all_df

amr_jaccard_shrinked=remove_genes(amr_jaccard,amr_clusters,amr_represent)
vf_jaccard_shrinked=remove_genes(vf_jaccard,vf_clusters,vf_represent)

amr_simpson_shrinked=remove_genes(amr_simpson,amr_clusters,amr_represent)
vf_simpson_shrinked=remove_genes(vf_simpson,vf_clusters,vf_represent)


# =============================================================================
# Multiple Testing (FDR) and q-value Categorization
# =============================================================================

def fdr_correction(data):
    p_values=data['P-value'].to_numpy()
    q_values=stats.false_discovery_control(p_values,method='bh')
    data['q-value']=q_values
    return data

amr_fdr=fdr_correction(amr_simpson_shrinked)
vf_fdr=fdr_correction(vf_simpson_shrinked)

def q_categorization(data):
    """
    Assign a q-value category for each gene pair:
    - 2: q-value < 0.01
    - 1: q-value < 0.05
    - 0: not significant
    """
    categories =[
            data['q-value'] < 0.01,
            data['q-value'] < 0.05
		]
    choices =[2,1]
    data['q-category']=np.select(categories,choices,default=0)
    return data

# Apply q-value categorization
categorized_amr=q_categorization(amr_fdr)
categorized_vf=q_categorization(vf_fdr)

# Merge Jaccard distances with Simpson + q-category info
amr_jaccard_and_simpson=pd.merge(categorized_amr,amr_jaccard_shrinked[['Gene 1','Gene 2','Phylogroup','Jaccard distance']],on=['Gene 1','Gene 2','Phylogroup'],how='left')
vf_jaccard_and_simpson=pd.merge(categorized_vf,vf_jaccard_shrinked[['Gene 1','Gene 2','Phylogroup','Jaccard distance']],on=['Gene 1','Gene 2','Phylogroup'],how='left')

amr_jaccard_and_simpson.to_csv(OUTPUT_DIR_1 / "amr_gene_jaccard_and_simpson_distances_2.csv")
vf_jaccard_and_simpson.to_csv(OUTPUT_DIR_1 / "vf_gene_jaccard_and_simpson_distances_2.csv")

# =============================================================================
# Determine Gene Order for Plotting
# =============================================================================

# Determine gene order for plotting by frequency within each phylogroup
for phylogroup in phylogroups:
    data_phylogroup=categorized_amr[categorized_amr['Phylogroup']==phylogroup]
    gene1_list=set(data_phylogroup['Gene 1'])
    gene2_list=set(data_phylogroup['Gene 2'])
    amr_df_phylogroup = amr_df[amr_df.index.isin(phylogroup_df[phylogroup_df['Phylogroup']==phylogroup].index)]
    sorted_genes = amr_df_phylogroup.sum().sort_values(ascending=False).index.tolist()
    gene1_sorted = [gene for gene in sorted_genes if gene in gene1_list]
    gene2_sorted = [gene for gene in sorted_genes if gene in gene2_list]
    print(f'Gene 1 {phylogroup}:{gene1_sorted}')
    print(f'Gene 2 {phylogroup}:{gene2_sorted}')

for phylogroup in phylogroups:
    data_phylogroup=categorized_vf[categorized_vf['Phylogroup']==phylogroup]
    gene1_list=set(data_phylogroup['Gene 1'])
    gene2_list=set(data_phylogroup['Gene 2'])
    vf_df_phylogroup = vf_df[vf_df.index.isin(phylogroup_df[phylogroup_df['Phylogroup']==phylogroup].index)]
    sorted_genes = vf_df_phylogroup.sum().sort_values(ascending=False).index.tolist()
    gene1_sorted = [gene for gene in sorted_genes if gene in gene1_list]
    gene2_sorted = [gene for gene in sorted_genes if gene in gene2_list]
    print(f'Gene 1 {phylogroup}:{gene1_sorted}')
    print(f'Gene 2 {phylogroup}:{gene2_sorted}')

def identical_number_genes(gene_dict):
    """"
    Group genes that have the same total count for reordering.
    Returns a list of gene groups with identical frequencies
    """
    value_to_genes = defaultdict(list)
    for gene, val in gene_dict.items():
        value_to_genes[val].append(gene)
    result =[genes for genes in value_to_genes.values() if len(genes) >1]
    return result

def sort_identical_number_gene1(data,target_genes):
    target = data[data['Gene 1'].isin(target_genes)]
    gene1_counts = target['Gene 1'].value_counts()
    gene1_order = gene1_counts.sort_values(ascending=True).index.tolist()
    return gene1_order

def sort_identical_number_gene2(data,target_genes):
    target = data[data['Gene 2'].isin(target_genes)]
    gene2_counts = target['Gene 2'].value_counts()
    gene2_order = gene2_counts.sort_values(ascending=False).index.tolist()
    return gene2_order


def reorder_by_parts(original_order, parts):
    """""
    Reorder genes while preserving block structure for identical frequency groups.
    """""
    i = 0
    result = []
    while i < len(original_order):
        matched = False
        for part in parts:
            if original_order[i] in part:
                block = []
                while i < len(original_order) and original_order[i] in part:
                    block.append(original_order[i])
                    i += 1
                block_sorted = [gene for gene in part if gene in block]
                result.extend(block_sorted)
                matched = True
                break
        if not matched:
            result.append(original_order[i])
            i += 1
    return result

def convert_gene_to_number(simpson_data, jaccard_data, gene_df):
    """""
    Convert gene names to numeric indices for heatmap plotting.
    Also creates a dictionary mapping indices back to gene names.
    """""
    all_simpson = pd.DataFrame()
    all_jaccard = pd.DataFrame()
    mapping_dict = {}

    for phylogroup in simpson_data['Phylogroup'].unique():
        simpson_phylogroup = simpson_data[simpson_data['Phylogroup'] == phylogroup]
        jaccard_phylogroup = jaccard_data[jaccard_data['Phylogroup'] == phylogroup]

        gene1_list = set(simpson_phylogroup['Gene 1'])
        gene2_list = set(simpson_phylogroup['Gene 2'])

        gene_df_phylogroup = gene_df[gene_df.index.isin(
            phylogroup_df[phylogroup_df['Phylogroup'] == phylogroup].index)]
        sorted_genes = gene_df_phylogroup.sum().sort_values(ascending=False).index.tolist()

        sorted_genes_dict = dict(gene_df_phylogroup.sum().sort_values(ascending=False))
        gene1_dict = {k: sorted_genes_dict[k] for k in gene1_list if k in sorted_genes_dict}
        gene2_dict = {k: sorted_genes_dict[k] for k in gene2_list if k in sorted_genes_dict}

        identical_number_gene1 = identical_number_genes(gene1_dict)
        identical_number_gene2 = identical_number_genes(gene2_dict)

        list_gene1 = []
        if identical_number_gene1:
            for genes in identical_number_gene1:
                list_gene1.append(sort_identical_number_gene1(simpson_phylogroup, genes))
        list_gene2 = []
        if identical_number_gene2:
            for genes in identical_number_gene2:
                list_gene2.append(sort_identical_number_gene2(simpson_phylogroup, genes))

        gene1_sorted = [gene for gene in sorted_genes if gene in gene1_list]
        gene2_sorted = [gene for gene in sorted_genes if gene in gene2_list]

        if list_gene1:
            gene1_sorted = reorder_by_parts(gene1_sorted, list_gene1)
        if list_gene2:
            gene2_sorted = reorder_by_parts(gene2_sorted, list_gene2)

        # Map to numeric indices
        dict_gene1 = dict(zip(gene1_sorted, range(len(gene1_sorted))))
        dict_gene2 = dict(zip(gene2_sorted, range(len(gene2_sorted))))

        # Reverse mapping
        index_to_gene1 = {v: k for k, v in dict_gene1.items()}
        index_to_gene2 = {v: k for k, v in dict_gene2.items()}

        mapping_dict[str(phylogroup)] = {
            'Gene 1': dict_gene1,
            'Gene 2': dict_gene2,
            'index_to_gene1': index_to_gene1,
            'index_to_gene2': index_to_gene2,
        }

        # Apply numeric conversion
        data_simpson = simpson_phylogroup.copy()
        data_simpson['Gene 1'] = data_simpson['Gene 1'].map(dict_gene1)
        data_simpson['Gene 2'] = data_simpson['Gene 2'].map(dict_gene2)

        data_jaccard = jaccard_phylogroup.copy()
        data_jaccard['Gene 1'] = data_jaccard['Gene 1'].map(dict_gene1)
        data_jaccard['Gene 2'] = data_jaccard['Gene 2'].map(dict_gene2)

        all_simpson = pd.concat([all_simpson, data_simpson])
        all_jaccard = pd.concat([all_jaccard, data_jaccard])

    return all_simpson, all_jaccard, mapping_dict

#Apply gene-to-number conversion for AMR and VF
numbered_amr_simpson,numbered_amr_jaccard,amr_dictionary=convert_gene_to_number(categorized_amr,amr_jaccard_shrinked,amr_df)
numbered_vf_simpson,numbered_vf_jaccard,vf_dictionary=convert_gene_to_number(categorized_vf,vf_jaccard_shrinked,vf_df)

#Merge converted Simpson and Jaccard distance tables
numbered_amr_data=pd.merge(numbered_amr_simpson,numbered_amr_jaccard[['Gene 1','Gene 2','Phylogroup','Jaccard distance']],on=['Gene 1','Gene 2','Phylogroup'],how='left')
numbered_vf_data =pd.merge(numbered_vf_simpson,numbered_vf_jaccard[['Gene 1','Gene 2','Phylogroup','Jaccard distance']],on=['Gene 1','Gene 2','Phylogroup'],how='left')


# Define a custom colormap (soft red-blue palette)
base_cmap = plt.get_cmap('RdYlBu_r')
soft_rdyblu = mcolors.LinearSegmentedColormap.from_list(
    'soft_rdyblu',
    base_cmap(np.linspace(0.15, 0.85, 100))
)

# =============================================================================
#  LaTeX Labels (AMR)
# =============================================================================
# These components are used to generate the AMR triangle heatmap.

label_amr_dict = {
    'aac(3)-IId': r"$\mathit{aac(3)\text{-}IId}$",
    'aadA2': r"$\mathit{aadA2}$",
    'aadA5': r"$\mathit{aadA5}$",
    'aph(6)-Id': r"$\mathit{aph(6)\text{-}Id}$",
    'blaCMY-2': r"$\mathit{bla}_{\mathrm{CMY\text{-}2}}$",
    'blaCTX-M-15': r"$\mathit{bla}_{\mathrm{CTX\text{-}M\text{-}15}}$",
    'blaOXA-1': r"$\mathit{bla}_{\mathrm{OXA\text{-}1}}$",
    'blaTEM-1': r"$\mathit{bla}_{\mathrm{TEM\text{-}1}}$",
    'ble': r"$\mathit{ble}$",
    'cyaA_S352T': r"$\mathit{cyaA}$ S352T",
    'dfrA17': r"$\mathit{dfrA17}$",
    'erm(B)': r"$\mathit{erm(B)}$",
    'ftsI_N337NYRIN': r"$\mathit{ftsI}$ N337NYRIN",
    'glpT_E448K': r"$\mathit{glpT}$ E448K",
    'gyrA_D87N': r"$\mathit{gyrA}$ D87N",
    'gyrA_S83L': r"$\mathit{gyrA}$ S83L",
    'marR_S3N': r"$\mathit{marR}$ S3N",
    'mph(A)': r"$\mathit{mph(A)}$",
    'parC_E84V': r"$\mathit{parC}$ E84V",
    'parC_S80I': r"$\mathit{parC}$ S80I",
    'parE_S458A': r"$\mathit{parE}$ S458A",
    'pmrB_E123D': r"$\mathit{pmrB}$ E123D",
    'ptsI_V25I': r"$\mathit{ptsI}$ V25I",
    'qnrS1': r"$\mathit{qnrS1}$",
    'sul1': r"$\mathit{sul1}$",
    'sul2': r"$\mathit{sul2}$",
    'tet(A)': r"$\mathit{tet(A)}$",
    'tet(B)': r"$\mathit{tet(B)}$",
    'uhpT_E350Q': r"$\mathit{uhpT}$ E350Q"
}

# =============================================================================
# AMR Triangle Heatmap
# =============================================================================

# Global plotting parameters
MARKER_SIZE = 12700    # Base marker size for squares
TICK_FONTSIZE = 62     # Font size for tick labels
FDR_SIZE = 200         # Marker size for FDR significance symbols


def make_amr_scatter(ax, data, gene1, gene2, max_gene2_length):
    """
    Draw a scatter plot (triangle heatmap) for AMR gene associations
    within a phylogroup.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis object to draw the scatter plot on.
    data : pd.DataFrame
        Data containing 'Gene 1', 'Gene 2', 'Simpson distance',
        'Jaccard distance', 'Odds ratio', and 'q-category'.
    gene1 : list
        Ordered list of representative genes for the y-axis.
    gene2 : list
        Ordered list of representative genes for the x-axis.
    max_gene2_length : int
        Maximum number of genes in gene2 (to align subplot widths).
    """

    # Set axis limits and remove spines
    x = len(gene2)
    y = len(gene1)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(-0.5, max_gene2_length - 0.5)
    ax.set_ylim(-0.5, (x - 1) + 0.5)

    # Replace gene names with LaTeX-styled labels (italic)
    gene2_labels = [label_amr_dict.get(g, g) for g in gene2]
    gene1_labels = [label_amr_dict.get(g, g) for g in gene1]

    # Set axis ticks and labels
    ax.set_xticks(np.arange(x))
    ax.set_yticks(np.arange(y))
    ax.set_xticklabels(gene2_labels, size=TICK_FONTSIZE, rotation=90)
    ax.set_yticklabels(gene1_labels, size=TICK_FONTSIZE)
    ax.invert_yaxis()

    # Set font style and color for axis tick labels
    for label_obj in ax.get_xticklabels():
        label_obj.set_color('black')
        label_obj.set_fontname("Arial")
    for label_obj in ax.get_yticklabels():
        label_obj.set_color('black')
        label_obj.set_fontname("Arial")

    # Plot squares by Simpson distance bin (size varies by distance range)
    for cutoff, size_rate in zip(
        [0.025, 0.05, 0.075, 0.1, 1],
        [1, 0.58, 0.23, 0.07, 0.015]
    ):
        if cutoff == 0.025:
            d = data[data['Simpson distance'] <= cutoff]
        elif cutoff == 1:
            d = data[data['Simpson distance'] > 0.1]
        else:
            d = data[
                (data['Simpson distance'] <= cutoff)
                & (data['Simpson distance'] > cutoff - 0.025)
            ]

        if not d.empty:
            ax.scatter(
                d['Gene 2'], d['Gene 1'],
                s=[size_rate * MARKER_SIZE] * len(d),
                linewidths=0,
                cmap=soft_rdyblu,
                alpha=0.7,
                marker="s",
                c=d['Odds ratio'],
                norm=LogNorm(vmin=0.01, vmax=100)
            )

    # Highlight thick (≤0.2) and thin (>0.2) Jaccard distance associations
    thick = data[data['Jaccard distance'] <= 0.2]
    thin = data[data['Jaccard distance'] > 0.2]

    if not thick.empty:
        ax.scatter(
            thick['Gene 2'], thick['Gene 1'],
            s=MARKER_SIZE,
            alpha=1,
            marker='s',
            facecolors='none',
            linewidths=4.5,
            edgecolors='black'
        )
    if not thin.empty:
        ax.scatter(
            thin['Gene 2'], thin['Gene 1'],
            s=MARKER_SIZE,
            alpha=1,
            marker='s',
            facecolors='none',
            linewidths=0.6,
            edgecolors='darkgrey'
        )

    # Overlay significance markers (+) for FDR thresholds
    for el1, el2 in zip([1, 2], [FDR_SIZE, FDR_SIZE * 7]):
        d = data[data['q-category'] == el1]
        ax.scatter(
            d['Gene 2'], d['Gene 1'],
            s=[el2] * len(d),
            marker="+",
            color='black'
        )

    return fig, ax


# Select phylogroups that are present in the dictionary
valid_phylogroups = [pg for pg in phylogroups if pg in amr_dictionary]

# Count the maximum number of unique gene2 values across phylogroups
gene2_counts = numbered_amr_data.groupby('Phylogroup')['Gene 2'].nunique()
max_gene2_count = gene2_counts[gene2_counts.index.isin(valid_phylogroups)].max()

# Determine subplot height ratios based on the number of gene1 in each phylogroup
height_ratios = [len(amr_dictionary[pg]['Gene 1']) for pg in valid_phylogroups]

# Create subplots (one row per phylogroup)
fig, axes = plt.subplots(
    len(valid_phylogroups), 1,
    figsize=(max_gene2_count * 2.1, sum(height_ratios) * 2.3),
    gridspec_kw={"height_ratios": height_ratios}
)
fig.subplots_adjust(hspace=2.5)

# Ensure axes is iterable even if there is only one subplot
if len(valid_phylogroups) == 1:
    axes = [axes]

# Loop over phylogroups and draw the scatter plots
for ax, phylogroup in zip(axes, valid_phylogroups):
    numbered_amr_phylogroup = numbered_amr_data[
        numbered_amr_data['Phylogroup'] == phylogroup
    ]
    dict_gene1 = amr_dictionary[phylogroup]['index_to_gene1']
    dict_gene2 = amr_dictionary[phylogroup]['index_to_gene2']

    gene1_list = [dict_gene1[i] for i in range(len(dict_gene1))]
    gene2_list = [dict_gene2[i] for i in range(len(dict_gene2))]

    make_amr_scatter(ax, numbered_amr_phylogroup, gene1_list, gene2_list, max_gene2_count)

# Adjust overall layout and save figure
fig.subplots_adjust(left=0.05, right=0.95)
plt.tight_layout()
plt.savefig(OUTPUT_DIR_2 / "triangle_heatmap_amr.pdf", dpi=600, bbox_inches='tight')

# =============================================================================
# LaTeX Labels (VF)
# =============================================================================
# These components are used to generate the VF triangle heatmap.

label_vf_dict=label_dict = {
    'cesD2': r"$\mathit{cesD2}$",
    'chuT': r"$\mathit{chuT}$",
    'chuW': r"$\mathit{chuW}$",
    'chuX': r"$\mathit{chuX}$",
    'chuY': r"$\mathit{chuY}$",
    'entC': r"$\mathit{entC}$",
    'escV': r"$\mathit{escV}$",
    'espA': r"$\mathit{espA}$",
    'espG': r"$\mathit{espG}$",
    'espK': r"$\mathit{espK}$",
    'espL1': r"$\mathit{espL1}$",
    'espL2': r"$\mathit{espL2}$",
    'espL4': r"$\mathit{espL4}$",
    'espR1': r"$\mathit{espR1}$",
    'espR3': r"$\mathit{espR3}$",
    'espX1': r"$\mathit{espX1}$",
    'espX2': r"$\mathit{espX2}$",
    'espX6': r"$\mathit{espX6}$",
    'espY1': r"$\mathit{espY1}$",
    'espY2': r"$\mathit{espY2}$",
    'espY4': r"$\mathit{espY4}$",
    'fepG': r"$\mathit{fepG}$",
    'fimA': r"$\mathit{fimA}$",
    'fimB': r"$\mathit{fimB}$",
    'fimI': r"$\mathit{fimI}$",
    'gspG': r"$\mathit{gspG}$",
    'gspL': r"$\mathit{gspL}$",
    'hlyD': r"$\mathit{hlyD}$",
    'iroB': r"$\mathit{iroB}$",
    'irp2': r"$\mathit{irp2}$",
    'iucB': r"$\mathit{iucB}$",
    'kpsD': r"$\mathit{kpsD}$",
    'kpsM': r"$\mathit{kpsM}$",
    'kpsT': r"$\mathit{kpsT}$",
    'nleE': r"$\mathit{nleE}$",
    'nleF': r"$\mathit{nleF}$",
    "nleG7'": r"$\mathit{nleG7'}$",
    'nleH1': r"$\mathit{nleH1}$",
    'papB': r"$\mathit{papB}$",
    'papD': r"$\mathit{papD}$",
    'papF': r"$\mathit{papF}$",
    'papG': r"$\mathit{papG}$",
    'papX': r"$\mathit{papX}$",
    'sepL': r"$\mathit{sepL}$",
    'sepQ/escQ': r"$\mathit{sepQ/escQ}$",
    'sfaC': r"$\mathit{sfaC}$",
    'shuA': r"$\mathit{shuA}$",
    'shuS': r"$\mathit{shuS}$",
    'shuT': r"$\mathit{shuT}$",
    'shuX': r"$\mathit{shuX}$",
    'shuY': r"$\mathit{shuY}$",
    'stx1B': r"$\mathit{stx1B}$",
    'stx2A': r"$\mathit{stx2A}$",
    'stxA': r"$\mathit{stxA}$",
    'yagV/ecpE': r"$\mathit{yagV/ecpE}$",
    'ybtA': r"$\mathit{ybtA}$"
}

# =============================================================================
# VF Triangle Heatmap
# =============================================================================

# Global plotting parameters for VF heatmaps
MARKER_SIZE = 14700    # Base marker size for squares
TICK_FONTSIZE = 62     # Font size for axis tick labels
FDR_SIZE = 300         # Marker size for FDR significance symbols


def make_vf_scatter(ax, data, gene1, gene2, max_gene2_length):
    """
    Draw a scatter plot (triangle heatmap) for VF gene associations
    within a phylogroup.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis object to draw the scatter plot on.
    data : pd.DataFrame
        Data containing 'Gene 1', 'Gene 2', 'Simpson distance',
        'Jaccard distance', 'Odds ratio', and 'q-category'.
    gene1 : list
        Ordered list of representative genes for the y-axis.
    gene2 : list
        Ordered list of representative genes for the x-axis.
    max_gene2_length : int
        Maximum number of genes in gene2 (to align subplot widths).
    """

    # Set axis limits and remove spines
    x = len(gene2)
    y = len(gene1)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(-0.5, max_gene2_length - 0.5)
    ax.set_ylim(-0.5, (x - 1) + 0.5)

    # Replace gene names with LaTeX-styled labels (italic)
    gene2_labels = [label_vf_dict.get(g, g) for g in gene2]
    gene1_labels = [label_vf_dict.get(g, g) for g in gene1]

    # Set axis ticks and labels
    ax.set_xticks(np.arange(0, y, 1))
    ax.set_yticks(np.arange(0, x, 1))
    ax.set_xticklabels(gene2_labels, size=TICK_FONTSIZE, rotation=90)
    ax.set_yticklabels(gene1_labels, size=TICK_FONTSIZE)
    ax.invert_yaxis()

    # Set font style and color for tick labels
    for label_obj in ax.get_xticklabels():
        label_obj.set_color('black')
        label_obj.set_fontname("Arial")
    for label_obj in ax.get_yticklabels():
        label_obj.set_color('black')
        label_obj.set_fontname("Arial")

    # Plot squares by Simpson distance bin (size varies by distance range)
    for cutoff, size_rate in zip(
        [0.025, 0.05, 0.075, 0.1, 1],
        [1, 0.58, 0.23, 0.07, 0.015]
    ):
        if cutoff == 0.025:
            d = data[data['Simpson distance'] <= cutoff]
        elif cutoff == 1:
            d = data[data['Simpson distance'] > 0.1]
        else:
            d = data[
                (data['Simpson distance'] <= cutoff)
                & (data['Simpson distance'] > cutoff - 0.025)
            ]

        if not d.empty:
            ax.scatter(
                d['Gene 2'], d['Gene 1'],
                s=[size_rate * MARKER_SIZE] * len(d),
                linewidths=0,
                cmap=soft_rdyblu,
                alpha=0.7,
                marker="s",
                c=d['Odds ratio'],
                norm=LogNorm(vmin=0.01, vmax=100)
            )

    # Highlight thick (≤0.2) and thin (>0.2) Jaccard distance associations
    thick = data[data['Jaccard distance'] <= 0.2]
    thin = data[data['Jaccard distance'] > 0.2]

    if not thick.empty:
        ax.scatter(
            thick['Gene 2'], thick['Gene 1'],
            s=MARKER_SIZE,
            alpha=1,
            marker='s',
            facecolors='none',
            linewidths=4.5,
            edgecolors='black'
        )
    if not thin.empty:
        ax.scatter(
            thin['Gene 2'], thin['Gene 1'],
            s=MARKER_SIZE,
            alpha=1,
            marker='s',
            facecolors='none',
            linewidths=0.6,
            edgecolors='darkgrey'
        )

    # Overlay significance markers (+) for FDR thresholds
    for el1, el2 in zip([1, 2], [FDR_SIZE, FDR_SIZE * 10]):
        d = data[data['q-category'] == el1]
        ax.scatter(
            d['Gene 2'], d['Gene 1'],
            s=[el2] * len(d),
            marker="+",
            color='black'
        )

    return fig, ax


# Select phylogroups that are present in the VF dictionary
valid_phylogroups = [pg for pg in phylogroups if pg in vf_dictionary]

# Count the maximum number of unique gene2 values across phylogroups
gene2_counts = numbered_vf_data.groupby('Phylogroup')['Gene 2'].nunique()
max_gene2_count = gene2_counts[gene2_counts.index.isin(valid_phylogroups)].max()

# Determine subplot height ratios based on the number of gene1 in each phylogroup
height_ratios = [len(vf_dictionary[pg]['Gene 1']) for pg in valid_phylogroups]

# Create subplots (one row per phylogroup)
fig, axes = plt.subplots(
    len(valid_phylogroups), 1,
    figsize=(max_gene2_count * 2.0, sum(height_ratios) * 2.2),
    gridspec_kw={"height_ratios": height_ratios}
)
fig.subplots_adjust(hspace=2.7)

# Ensure axes is iterable even if there is only one subplot
if len(valid_phylogroups) == 1:
    axes = [axes]

# Loop over phylogroups and draw the scatter plots
for ax, phylogroup in zip(axes, valid_phylogroups):
    numbered_vf_phylogroup = numbered_vf_data[
        numbered_vf_data['Phylogroup'] == phylogroup
    ]
    dict_gene1 = vf_dictionary[phylogroup]['index_to_gene1']
    dict_gene2 = vf_dictionary[phylogroup]['index_to_gene2']

    gene1_list = [dict_gene1[i] for i in range(len(dict_gene1))]
    gene2_list = [dict_gene2[i] for i in range(len(dict_gene2))]

    make_vf_scatter(ax, numbered_vf_phylogroup, gene1_list, gene2_list, max_gene2_count)

# Adjust overall layout and save figure
fig.subplots_adjust(left=0.05, right=0.95)
plt.tight_layout()
plt.savefig(OUTPUT_DIR_2 / "triangle_heatmap_vf.pdf", dpi=600, bbox_inches='tight')


# =============================================================================
# Standalone Legend Figure (Simpson, q-value/FDR, Odds Ratio, Jaccard)
# =============================================================================

# Select a strong red color from the custom colormap
most_red = soft_rdyblu(1.0)

# --- Figure initialization ---
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

# --- Parameter settings ---
simpson_bins = [
    'SD ≤ 0.025', '0.025 < SD ≤ 0.05',
    '0.05 < SD ≤ 0.075', '0.075 < SD ≤ 0.1',
    'SD > 0.1'
]
simpson_sizes = [1, 0.58, 0.23, 0.07, 0.015]
MARKER_SIZE_BASE = 5500
FDR_SIZE = 150
FDR_OUTER_SIZE = 5500

# --- 1. Simpson distance legend (red squares) ---
# Position markers vertically from top (7.8) to bottom (1.8)
y_positions = np.linspace(7.8, 1.8, 5)

for i, (label, inner_rate) in enumerate(zip(simpson_bins, simpson_sizes)):
    y = y_positions[i]
    # Inner filled square (scaled by Simpson distance bin)
    ax.scatter(
        0.8, y, s=MARKER_SIZE_BASE * inner_rate,
        color=most_red, alpha=0.8, marker='s', linewidths=0
    )
    # Outer square outline
    ax.scatter(
        0.8, y, s=MARKER_SIZE_BASE,
        facecolors='none', edgecolors='darkgrey',
        marker='s', linewidths=0.6
    )
    ax.text(1.6, y, label, ha='left', va='center', fontsize=22)

ax.text(0.3, 8.8, "Simpson distance", fontsize=32, fontweight='bold')
ax.text(0.3, 0.2, "SD : Simpson distance", fontsize=28)
ax.text(7, 0.2, "JD : Jaccard distance", fontsize=28)

# --- 2. FDR significance legend (+ symbols) ---
fdr_labels = ['FDR < 0.05', 'FDR < 0.01']
fdr_sizes = [FDR_SIZE, FDR_SIZE * 10]
fdr_y_positions = [3.3, 1.8]

for y, lbl, sz in zip(fdr_y_positions, fdr_labels, fdr_sizes):
    # Grey square outline
    ax.scatter(
        7, y, s=FDR_OUTER_SIZE,
        facecolors='none', edgecolors='darkgrey',
        marker='s', linewidths=0.6
    )
    # Black plus sign
    ax.scatter(
        7, y, s=sz, color='black',
        marker='+', linewidths=0.6
    )
    ax.text(7.9, y, lbl, ha='left', va='center', fontsize=22)

ax.text(6.3, 4.3, "q-value", fontsize=32, fontweight='bold')

# --- 3. Odds Ratio colorbar (vertical) ---
cax = fig.add_axes([0.86, 0.38, 0.05, 0.5])  # [left, bottom, width, height]
cb = plt.colorbar(
    plt.cm.ScalarMappable(norm=LogNorm(vmin=0.01, vmax=100), cmap=soft_rdyblu),
    cax=cax, orientation='vertical'
)
cb.ax.tick_params(labelsize=20)

ax.text(
    13.5, 8.8, "Odds Ratio",
    fontsize=32, fontweight='bold',
    rotation=0, ha='center'
)

# --- 4. Jaccard distance legend (thick vs thin outlines) ---
jaccard_labels = ['JD ≤ 0.2', 'JD > 0.2']
jaccard_linewidths = [4.5, 0.6]  # Thick and thin line widths
jaccard_y_positions = [7.8, 6.3]
colors = ['black', 'darkgrey']

for y, lbl, lw, cl in zip(jaccard_y_positions, jaccard_labels, jaccard_linewidths, colors):
    ax.scatter(
        7, y, s=MARKER_SIZE_BASE,
        facecolors='none', edgecolors=cl,
        marker='s', linewidths=lw
    )
    ax.text(7.8, y, lbl, ha='left', va='center', fontsize=22)

ax.text(6.3, 8.8, "Jaccard distance", fontsize=32, fontweight='bold')

# --- Final layout and save ---
ax.set_ylim(0, 9)
ax.set_xlim(0, 12)
fig.subplots_adjust(top=0.95, bottom=0.05, left=0.1, right=0.85)

fig.savefig(
    OUTPUT_DIR_2 / "legend_for_triangle_heatmap.pdf",
    bbox_inches='tight', dpi=300
)
