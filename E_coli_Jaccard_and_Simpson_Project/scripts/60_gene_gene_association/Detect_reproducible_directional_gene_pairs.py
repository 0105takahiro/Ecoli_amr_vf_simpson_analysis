# -*- coding: utf-8 -*-
"""
Reproducible directional gene pairs across phylogroups and major STs.

Pipeline outline
----------------
1) Load metadata (phylogroup, ST) and gene presence/absence matrices (AMR, VF).
2) From precomputed gene–gene distance tables (Simpson/Jaccard + q-values),
   select pairs meeting the directional association criteria per phylogroup:
      Simpson distance ≤ 0.1, Jaccard distance > 0.2, q-value < 0.05
3) For each eligible (gene1, gene2, phylogroup), find the top-5 STs in which
   gene1 is present (within the phylogroup), then compute:
      - 2×2 Fisher’s exact test p-value and odds ratio (via Table2x2)
      - Jaccard distance
      - Simpson distance
4) Apply Benjamini–Hochberg FDR to the new ST-level p-values.
5) Retain pairs that meet the same criteria in ≥2 STs within the same phylogroup.
6) Output a summary table.

"""

from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np

from sklearn.metrics import jaccard_score
from py_stringmatching import OverlapCoefficient
from statsmodels.stats.contingency_tables import Table2x2
from statsmodels.stats.multitest import multipletests

# ----------------------------
# Configuration
# ----------------------------
ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR_1 = ROOT / "Output" / "Preparation"
INPUT_DIR_2 = ROOT / "Output" / "AMR_and_VF_genes_ok"
INPUT_DIR_3 = ROOT / "Output" / "gene_gene_distance"

OUTPUT_DIR = ROOT / "Output" / "reproducible_directional_gene_pairs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIGURES_DIR = ROOT / "Figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PHYLOGROUP_CSV   = INPUT_DIR_1 / "ecoli-genomes_filtered_25080_phylogroup.csv"
ST_CSV           = INPUT_DIR_1 / "ecoli-genomes_filtered_25080_MLST.csv"
AMR_CSV          = INPUT_DIR_2 / "amr-genes-presence-absence.csv"
VF_CSV           = INPUT_DIR_2 / "vf-genes-presence-absence.csv"
AMR_DISTANCE_CSV = INPUT_DIR_3 / "amr_gene_jaccard_and_simpson_distances.csv"
VF_DISTANCE_CSV  = INPUT_DIR_3 / "vf_gene_jaccard_and_simpson_distances.csv"

# Analysis constants
PHYLOGROUPS = ['A', 'B1', 'B2', 'C', 'D', 'E', 'F', 'G']

# ----------------------------
# Load inputs
# ----------------------------
phylogroup_df = pd.read_csv(PHYLOGROUP_CSV, index_col=0)  # index: genome, col: 'Phylogroup'
st_df         = pd.read_csv(ST_CSV, index_col=0)          # index: genome, col: 'ST'
amr_df        = pd.read_csv(AMR_CSV, index_col=0)         # presence/absence (rows=genomes, cols=genes)
vf_df         = pd.read_csv(VF_CSV,  index_col=0)         # presence/absence (rows=genomes, cols=genes)
amr_distance  = pd.read_csv(AMR_DISTANCE_CSV, index_col=0)
vf_distance   = pd.read_csv(VF_DISTANCE_CSV,  index_col=0)

# String matcher for Simpson similarity (Overlap Coefficient)
oc = OverlapCoefficient()


def _compute_pair_stats_in_top_sts(
    eligible_pairs: pd.DataFrame,
    gene_matrix: pd.DataFrame,
    phylogroup: str
) -> pd.DataFrame:
    """
    For the given phylogroup and eligible (gene1, gene2) pairs, find top-5 STs
    (by counts of gene1-positive genomes within the phylogroup) and compute
    association metrics per ST.

    Returns a long-format DataFrame with columns:
        ['Gene 1','Gene 2','ST','No. of genomes of the ST',
         'No. of genomes with both genes','No. of genomes with gene 1',
         'No. of genomes with gene 2','Jaccard distance','Simpson distance',
         'p-value','odds-ratio','Phylogroup']
    """
    out_rows = []

    # genomes belonging to the current phylogroup
    pg_genomes = phylogroup_df.index[phylogroup_df['Phylogroup'] == phylogroup]
    pg_gene_mat = gene_matrix.loc[gene_matrix.index.intersection(pg_genomes)]

    for gene1, gene2 in zip(eligible_pairs['Gene 1'], eligible_pairs['Gene 2']):
        # ST distribution among gene1-positive genomes in this phylogroup
        gene1_positive_idx = pg_gene_mat.index[pg_gene_mat[gene1] == 1]
        st_sub = st_df.loc[st_df.index.intersection(gene1_positive_idx)]
        st_counts = Counter(st_sub['ST'])
        # drop missing STs represented by '-' (if present)
        if '-' in st_counts:
            del st_counts['-']

        # top5 STs
        top_sts = [st for st, _ in sorted(st_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]]

        for st in top_sts:
            # genomes in this ST within current phylogroup
            st_idx = st_df.index[(st_df['ST'] == st) & (st_df.index.isin(pg_genomes))]
            sub = pg_gene_mat.loc[pg_gene_mat.index.intersection(st_idx)]
            if sub.empty:
                continue

            # 2x2 contingency table and Fisher's exact test via Table2x2
            x = sub[gene1]
            y = sub[gene2]
            contingency = pd.crosstab(x, y).reindex(index=[0, 1], columns=[0, 1], fill_value=0).astype(int)
            tbl = Table2x2(contingency.values)
            odds_ratio = tbl.oddsratio
            p_value = tbl.test_nominal_association().pvalue

            # Jaccard distance on binary presence vectors
            jacc_dist = 1.0 - jaccard_score(x, y)

            # Simpson distance via Overlap Coefficient on positive index sets
            pos1 = x.index[x == 1].tolist()
            pos2 = y.index[y == 1].tolist()
            simpson_sim = oc.get_sim_score(pos1, pos2)
            simpson_dist = 1.0 - simpson_sim

            out_rows.append({
                'Gene 1': gene1,
                'Gene 2': gene2,
                'ST': f'ST{st}',
                'No. of genomes of the ST': int(len(sub)),
                'No. of genomes with both genes': int((x.eq(1) & y.eq(1)).sum()),
                'No. of genomes with gene 1': int(x.sum()),
                'No. of genomes with gene 2': int(y.sum()),
                'Jaccard distance': float(jacc_dist),
                'Simpson distance': float(simpson_dist),
                'p-value': float(p_value),
                'odds-ratio': float(odds_ratio),
                'Phylogroup': phylogroup
            })

    return pd.DataFrame(out_rows)


def _expand_pairs_by_st(distance_df: pd.DataFrame, gene_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    For each phylogroup, select directional pairs by the pre-specified criteria
    from the gene–gene distance table and compute per-ST association stats.
    """
    rows = []
    for pg in PHYLOGROUPS:
        eligible = distance_df[
            (distance_df['Phylogroup'] == pg) &
            (distance_df['Simpson distance'] <= 0.1) &
            (distance_df['Jaccard distance'] > 0.2) &
            (distance_df['q-value'] < 0.05)
        ]
        if eligible.empty:
            continue
        rows.append(_compute_pair_stats_in_top_sts(eligible, gene_matrix, pg))

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=[
        'Gene 1', 'Gene 2', 'ST',
        'No. of genomes of the ST', 'No. of genomes with both genes',
        'No. of genomes with gene 1', 'No. of genomes with gene 2',
        'Jaccard distance', 'Simpson distance', 'p-value', 'odds-ratio', 'Phylogroup'
    ])


def _apply_fdr_bh(df: pd.DataFrame, p_col: str = 'p-value', out_col: str = 'q-value') -> pd.DataFrame:
    """Apply Benjamini–Hochberg FDR correction to the p-values."""
    if df.empty:
        df[out_col] = []
        return df
    pvals = df[p_col].to_numpy()
    _, qvals, _, _ = multipletests(pvals, method='fdr_bh')
    df = df.copy()
    df[out_col] = qvals
    return df


def _filter_pairs_reproduced_across_sts(st_level_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep (gene1, gene2, phylogroup) that meet the criteria in >= 2 STs:
        Simpson distance ≤ 0.1, q-value < 0.05, Jaccard distance > 0.2
    """
    if st_level_df.empty:
        return st_level_df

    keep_rows = []
    # group by pair and phylogroup
    for (g1, g2, pg), grp in st_level_df.groupby(['Gene 1', 'Gene 2', 'Phylogroup']):
        met = grp[
            (grp['Simpson distance'] <= 0.1) &
            (grp['q-value'] < 0.05) &
            (grp['Jaccard distance'] > 0.2)
        ]
        if len(met) >= 2:
            keep_rows.append(met)

    return pd.concat(keep_rows, ignore_index=True) if keep_rows else pd.DataFrame(columns=st_level_df.columns)


def _summarize_pairs_by_phylogroup_st(st_filtered_df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a wide table:
        columns = ['Gene 1', 'Gene 2'] + PHYLOGROUPS
        each cell lists STs (comma-separated) in which the pair reproduced within the phylogroup.
    """
    if st_filtered_df.empty:
        return pd.DataFrame(columns=['Gene 1', 'Gene 2'] + PHYLOGROUPS)

    rows = []
    unique_pairs = st_filtered_df[['Gene 1', 'Gene 2']].drop_duplicates()

    for gene1, gene2 in unique_pairs.itertuples(index=False):
        row = {'Gene 1': gene1, 'Gene 2': gene2}
        pair_df = st_filtered_df[(st_filtered_df['Gene 1'] == gene1) &
                                 (st_filtered_df['Gene 2'] == gene2)]
        for pg in PHYLOGROUPS:
            st_list = pair_df.loc[pair_df['Phylogroup'] == pg, 'ST'].tolist()
            # strip "ST" prefix for compactness (e.g., "131, 73")
            st_clean = [s.replace('ST', '') for s in st_list]
            row[pg] = ', '.join(st_clean)
        rows.append(row)

    cols = ['Gene 1', 'Gene 2'] + PHYLOGROUPS
    return pd.DataFrame(rows, columns=cols)


# ----------------------------
# Main analysis
# ----------------------------
# Expand AMR/VF pairs into per-ST stats under each phylogroup
amr_st_df = _expand_pairs_by_st(amr_distance, amr_df)
vf_st_df  = _expand_pairs_by_st(vf_distance,  vf_df)

# FDR (BH) on the new ST-level p-values
amr_st_df = _apply_fdr_bh(amr_st_df)
vf_st_df  = _apply_fdr_bh(vf_st_df)

# Keep pairs reproduced in >= 2 STs within the same phylogroup
amr_repro_df = _filter_pairs_reproduced_across_sts(amr_st_df)
vf_repro_df  = _filter_pairs_reproduced_across_sts(vf_st_df)

# Combine AMR + VF summary
combined_repro_df = pd.concat([amr_repro_df, vf_repro_df], ignore_index=True)
summary_by_pair = _summarize_pairs_by_phylogroup_st(combined_repro_df)

# Save summary
summary_by_pair.to_csv(OUTPUT_DIR/ "reproducible_gene_pairs.csv", index=False)

