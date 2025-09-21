import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path

# ----------------------------
# Configuration 
# ----------------------------
ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = ROOT / "output"
OUTPUT_DIR = ROOT / "output" / "04_fundamental_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILTERED_FILES_CSV = INPUT_DIR / "01_preparation"/ "ecoli_genomes_filtered_25080.csv"

METADATA_CSV = INPUT_DIR / "01_preparation"/ "ecoli_genomes_25875_metadata.csv"
ASSEMBLY_STATS_CSV = INPUT_DIR / "01_preparation" / "ecoli_genomes_25875_assembly_stats.csv"

PHYLOGROUP_CSV = INPUT_DIR / "01_preparation" /"ecoli_genomes_filtered_25080_phylogroup.csv"
ST_CSV = INPUT_DIR / "01_preparation" / "ecoli_genomes_filtered_25080_mlst.csv"
AMR_CSV = INPUT_DIR / "02_gene_screening" / "amr_and_vf_genes" / "amr_genes_presence_absence.csv"
VF_CSV = INPUT_DIR / "02_gene_screening" /"amr_and_vf_genes" / "vf_genes_presence_absence.csv"

filtered_files_df = pd.read_csv(FILTERED_FILES_CSV,index_col=0)

metadata_df = pd.read_csv(METADATA_CSV, index_col=0)
assembly_df = pd.read_csv(ASSEMBLY_STATS_CSV, index_col=0)

phylogroup_df = pd.read_csv(PHYLOGROUP_CSV, index_col=0)
st_df = pd.read_csv(ST_CSV, index_col=0)
amr_df = pd.read_csv(AMR_CSV, index_col=0)
vf_df = pd.read_csv(VF_CSV, index_col=0)

metadata_df = metadata_df[metadata_df.index.isin(filtered_files_df.index)]
assembly_df = assembly_df[assembly_df.index.isin(filtered_files_df.index)]

metadata_df=pd.concat([metadata_df,phylogroup_df,st_df],axis=1)

# ----------------------------
# 2. Host distribution (Top 5 + Others)
# ----------------------------
host_counts = metadata_df['Host'].value_counts()
top5_hosts = host_counts.iloc[:5]
others_sum = host_counts.iloc[5:].sum()
host_summary = pd.concat([top5_hosts, pd.Series(others_sum, index=['Others'])])
host_summary = host_summary.to_frame(name='No. of Genomes')
host_summary.index.name = 'Host'

# ----------------------------
# 3. Format metadata columns
# ----------------------------

metadata_df = metadata_df[[
    'Organism', 'Submission date', 'Coverage', 'Assembly status',
    'BioSample accession', 'Host', 'Country', 'Isolation source',
    'Collection date', 'ST', 'Phylogroup'
]]

# ----------------------------
# 4. Assembly quality (N50, contig count)
# ----------------------------
N50_min, N50_med, N50_max = np.percentile(assembly_df['N50'], [0, 50, 100])
contig_min, contig_med, contig_max = np.percentile(assembly_df['Contig number'], [0, 50, 100])
print(f"N50 (assembly contiguity): min={N50_min:.0f}, median={N50_med:.0f}, max={N50_max:.0f}")
print(f"Contig count: min={contig_min:.0f}, median={contig_med:.0f}, max={contig_max:.0f}\n")

# ----------------------------
# 5. ST distribution
# ----------------------------
st_counts = metadata_df['ST'].value_counts()
print("Top 5 sequence types (ST) by frequency:")
print(st_counts.head(5), "\n")
singleton_ST_count = (st_counts == 1).sum()
print("Number of STs that appear in only one genome:", singleton_ST_count, "\n")

# ----------------------------
# 6. Phylogroup summary and gene profile diversity
# ----------------------------
phylogroup_list = ['A', 'B1', 'B2', 'C', 'D', 'E', 'F', 'G']
phylo_counts = metadata_df['Phylogroup'].value_counts().reindex(index=phylogroup_list)
phylo_counts.name = 'No. of Genomes'

def count_unique_profiles(data_df, subset_index):
    group_data = data_df[data_df.index.isin(subset_index)]
    return len(group_data.drop_duplicates())

unique_amr = {}
unique_vf = {}
for pg in phylogroup_list:
    pg_isolates = metadata_df[metadata_df['Phylogroup'] == pg].index
    unique_amr[pg] = count_unique_profiles(amr_df, pg_isolates)
    unique_vf[pg] = count_unique_profiles(vf_df, pg_isolates)

phylogroup_summary = pd.DataFrame({
    'No. of Genomes': phylo_counts,
    'No. of AMR Gene Profiles': pd.Series(unique_amr),
    'No. of VF Gene Profiles': pd.Series(unique_vf)
})
phylogroup_summary.index.name = 'Phylogroup'

# ----------------------------
# 7. Cross-tabulate phylogroup × Host category
# ----------------------------
target_hosts = ['homo sapiens', 'canis lupus familiaris', 'swine', 'bovine', 'chicken']

metadata_df['Host'] = metadata_df['Host'].astype(str).str.strip().str.lower()
metadata_df['Host category'] = metadata_df['Host'].apply(lambda x: x if x in target_hosts else 'Others')
host_by_phylogroup = pd.crosstab(metadata_df['Phylogroup'], metadata_df['Host category'])
host_by_phylogroup = host_by_phylogroup.reindex(columns=target_hosts + ['Others'], fill_value=0)

summary = pd.concat([host_by_phylogroup, phylogroup_summary], axis=1)
summary = summary.reindex(columns=[
    'No. of Genomes', 'homo sapiens', 'canis lupus familiaris',
    'swine', 'bovine', 'chicken', 'Others',
    'No. of AMR Gene Profiles', 'No. of VF Gene Profiles'
])

# ----------------------------
# 8. Add total row and save
# ----------------------------

cols_to_sum = summary.columns[:-2]
summary_totals = summary[cols_to_sum].sum()
n_amr_unique = amr_df.apply(lambda row: tuple(row), axis=1).nunique()
n_vf_unique = vf_df.apply(lambda row: tuple(row), axis=1).nunique()

total_row = pd.Series(
    list(summary_totals) + [n_amr_unique, n_vf_unique],
    index=summary.columns, name='Total'
)

summary_with_total = pd.concat([summary, pd.DataFrame([total_row])])
summary_with_total.columns=[
    'No. of Genomes', 'Homo sapiens', 'Canis lupus familiaris',
    'Swine', 'Bovine', 'Chicken', 'Others',
    'No. of AMR Gene Profiles', 'No. of VF Gene Profiles'
    ]
summary_with_total.to_csv(OUTPUT_DIR / "genome_characteristics.csv")

# ----------------------------
# 9. Distribution of gene counts per genome
# ----------------------------

# AMR
amr_gene_counts = amr_df.sum(axis=1)
amr_counts_df = pd.merge(amr_gene_counts.to_frame(name='No. of AMR genes'),
                         metadata_df[['Phylogroup']],
                         left_index=True, right_index=True)
print(f"AMR genes per genome: min={amr_counts_df['No. of AMR genes'].min()}, "
      f"median={np.median(amr_counts_df['No. of AMR genes']):.1f}, "
      f"max={amr_counts_df['No. of AMR genes'].max()}")
print("Distribution of genomes by AMR gene count:")
print(amr_counts_df['No. of AMR genes'].value_counts().sort_index().to_dict(), "\n")

# VF
vf_gene_counts = vf_df.sum(axis=1)
vf_counts_df = pd.merge(vf_gene_counts.to_frame(name='No. of VF genes'),
                        metadata_df[['Phylogroup']],
                        left_index=True, right_index=True)
print(f"VF genes per genome: min={vf_counts_df['No. of VF genes'].min()}, "
      f"median={np.median(vf_counts_df['No. of VF genes']):.1f}, "
      f"max={vf_counts_df['No. of VF genes'].max()}")
print("Distribution of genomes by VF gene count:")
print(vf_counts_df['No. of VF genes'].value_counts().sort_index().to_dict(), "\n")

# ----------------------------
# 10. Boxplot for AMR/VF gene counts per genome
# ----------------------------

import matplotlib
phylogroup_order = ['A', 'B1', 'B2', 'C', 'D', 'E', 'F', 'G']
color_palette = ["#4670ef", "#f2c475e7", "#6ce187d6", "#f7605a",
                 "#a367f1ef", "#b988593a", "#f89be5", "#bab8b886"]
palette = sns.color_palette(color_palette)

FIGURE_OUTPUT_DIR = ROOT/ "Figures"
FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIGURE = FIGURE_OUTPUT_DIR / "number_of_amr_vf_genes_per_genome.pdf"
plt.rcParams['font.family'] = 'Arial'

fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharex=True, gridspec_kw={'wspace': 0.4})

# Plot A: AMR
sns.boxplot(x='Phylogroup', y='No. of AMR genes', data=amr_counts_df,
            order=phylogroup_order, palette=palette, width=0.7, ax=axs[0], fliersize=0.1)
axs[0].set_ylabel('No. of AMR genes', fontsize=14)
axs[0].set_xlabel('')
axs[0].set_yticks(range(0, 41, 10))
axs[0].tick_params(axis='both', labelsize=14)
axs[0].set_ylim(-1, 42)
axs[0].set_xlim(-0.7, 7.7)

# Plot B: VF
sns.boxplot(x='Phylogroup', y='No. of VF genes', data=vf_counts_df,
            order=phylogroup_order, palette=palette, width=0.7, ax=axs[1], fliersize=0.1)
axs[1].set_ylabel('No. of VF genes', fontsize=14)
axs[1].set_xlabel('')
axs[1].set_yticks([0, 50, 100, 150])
axs[1].tick_params(axis='both', labelsize=14)
axs[1].set_ylim(-3, 140)
axs[1].set_xlim(-0.7, 7.7)

fig.supxlabel('Phylogroup', fontsize=20, y=-0.03)
fig.text(0.06, 0.9, 'a', fontsize=20)
fig.text(0.51, 0.9, 'b', fontsize=20)

plt.savefig(OUT_FIGURE, dpi=600, bbox_inches='tight')
