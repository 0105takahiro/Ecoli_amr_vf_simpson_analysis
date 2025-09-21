import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
from scipy.spatial.distance import squareform
from scipy.stats import sem, t
from tqdm import tqdm
from pathlib import Path
from matplotlib.lines import Line2D

# ----------------------------
# Figure / font settings
# ----------------------------
plt.rcParams['font.family'] = 'Arial'

ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = ROOT / "output"
GENOTYPIC_DISTANCE_DIR = INPUT_DIR / "genotypic_distance_matrix_each_phylogroup"
SNP_DISTANCE_DIR = INPUT_DIR / "snp_dists_output"

PHYLOGROUP_CSV = INPUT_DIR / "preparation" /"ecoli_genomes_filtered_25080_phylogroup.csv"
AMR_CSV = INPUT_DIR / "amr_and_vf_genes" / "amr_genes_presence_absence.csv"
VF_CSV = INPUT_DIR / "amr_and_vf_genes" / "vf_genes_presence_absence.csv"

FIGURE_OUTPUT_DIR = ROOT / "figures" / "ecdf"
FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Load base tables
# ----------------------------
amr_df        = pd.read_csv(AMR_CSV, index_col=0)
vf_df         = pd.read_csv(VF_CSV,  index_col=0)
phylogroup_df = pd.read_csv(PHYLOGROUP_CSV, index_col=0)

# Target phylogroups
phylogroups = ['A','B1','B2','C','D','E','F','G']

# ----------------------------
# Utility functions
# ----------------------------
def get_random_representatives(binary_df: pd.DataFrame, random_state=None):
    """
    Collapse identical genotypes and sample one representative per genotype cluster.
    Returns list of selected genome IDs (index labels).
    """
    # Build genotype label (string) without mutating caller's df
    genotype_labels = binary_df.astype(int).astype(str).agg(''.join, axis=1)
    groups = genotype_labels.groupby(genotype_labels)
    reps = (binary_df
            .assign(_geno=genotype_labels)
            .groupby('_geno', group_keys=False)
            .sample(n=1, random_state=random_state))
    return reps.index.tolist()

def extract_lower_percent_snp_pairs(snp_sq: pd.DataFrame,
                                    dist_sq: pd.DataFrame,
                                    genome_ids: list,
                                    percentile: float):
    """
    For a genome subset, flatten SNP & distance matrices (lower triangle),
    then keep distances whose SNP values are <= percentile threshold.
    Returns a 1D list of distances.
    """
    # Reindex to the subset and ensure order alignment
    snp_sub  = snp_sq.loc[genome_ids, genome_ids]
    dist_sub = dist_sq.loc[genome_ids, genome_ids]

    # To condensed vectors
    snp_flat  = squareform(snp_sub.values)
    dist_flat = squareform(dist_sub.values)

    # Percentile threshold on SNP
    thr = np.percentile(snp_flat, percentile)
    return [d for s, d in zip(snp_flat, dist_flat) if s <= thr]

def compute_ecdf_iterations(snp_sq: pd.DataFrame,
                            amr_jac: pd.DataFrame,
                            amr_sim: pd.DataFrame,
                            vf_jac: pd.DataFrame,
                            vf_sim: pd.DataFrame,
                            amr_bin: pd.DataFrame,
                            vf_bin: pd.DataFrame,
                            iterations: int = 1000,
                            percentile: float = 0.1,
                            progress_desc: str = ""):
    """
    Repeat: sample genotype-representatives → take cgSNP <= percentile pairs → collect distances.
    Returns dict of lists-of-lists for each metric key.
    """
    out = {'AMR Jaccard': [], 'AMR Simpson': [], 'VF Jaccard': [], 'VF Simpson': []}

    for _ in tqdm(range(iterations), desc=progress_desc):
        amr_reps = get_random_representatives(amr_bin)  # dedup AMR genotypes
        vf_reps  = get_random_representatives(vf_bin)   # dedup VF genotypes

        out['AMR Jaccard'].append(extract_lower_percent_snp_pairs(snp_sq, amr_jac, amr_reps, percentile))
        out['AMR Simpson'].append(extract_lower_percent_snp_pairs(snp_sq, amr_sim, amr_reps, percentile))
        out['VF Jaccard'].append(extract_lower_percent_snp_pairs(snp_sq, vf_jac, vf_reps, percentile))
        out['VF Simpson'].append(extract_lower_percent_snp_pairs(snp_sq, vf_sim, vf_reps, percentile))

    return out

def plot_ecdf_with_ci_fixed_x(dist_lists, label, color='blue', interval=0.0001, lw=3):
    """
    Make a mean ECDF curve across iterations with 95% CI (shaded).
    Approach:
      - For each iteration list 'd', sort distances
      - Interpolate quantiles onto a fixed grid x in [0,1)
      - Stack → mean and CI → plot (x-axis: distance, y-axis: cumulative probability)
    Returns: mean curve (list) sampled on fixed y-grid.
    """
    # Filter empty iterations to avoid NaNs
    dist_lists = [np.sort(np.asarray(d, dtype=float)) for d in dist_lists if len(d) > 0]
    if len(dist_lists) == 0:
        return []

    # y-grid (CDF) & interpolate inverse-CDF onto uniform y-grid
    y_vals = np.arange(0, 1, interval)
    # To avoid issues when len(d)==1, ensure xp strictly increasing
    interpolated = []
    for d in dist_lists:
        if len(d) == 1:
            interpolated.append(np.repeat(d[0], len(y_vals)))
        else:
            xp = np.linspace(0, 1, len(d), endpoint=False)  # positions for each sorted value
            interpolated.append(np.interp(y_vals, xp, d, left=d[0], right=d[-1]))

    data = np.vstack(interpolated)  # shape: (iterations, len(y_vals))
    mean  = np.nanmean(data, axis=0)
    stderr = sem(data, axis=0, nan_policy='omit')
    ci95  = stderr * t.ppf(0.975, df=data.shape[0]-1)

    # Plot (distance on X, CDF on Y)
    plt.plot(mean, y_vals, label=label, color=color, lw=lw)
    plt.fill_betweenx(y_vals, mean - ci95, mean + ci95, color=color, alpha=0.2)

    return mean.tolist()

# Colors per metric
LABEL_COLOR = {
    'JaccardAMR':  'red',
    'SimpsonAMR':  'orange',
    'JaccardVF':   'blue',
    'SimpsonVF':   'green'
}

# ----------------------------
# Main: per-phylogroup ECDF + summary stats
# ----------------------------
summary_rows = []  # collect numbers for manuscript table

for pg in phylogroups:
    pg_lower = pg.lower()
    # --- Load square matrices for this phylogroup ---
    snp_sq  = pd.read_csv(SNP_DISTANCE_DIR /  f"snp_dists_{pg_lower}.csv",index_col=0)
    amr_jac = pd.read_csv(GENOTYPIC_DISTANCE_DIR / f"amr_jaccard_distance_{pg_lower}.csv", index_col=0)
    amr_sim = pd.read_csv(GENOTYPIC_DISTANCE_DIR / f"amr_simpson_distance_{pg_lower}.csv",index_col=0)
    vf_jac  = pd.read_csv(GENOTYPIC_DISTANCE_DIR / f"vf_jaccard_distance_{pg_lower}.csv",index_col=0)
    vf_sim  = pd.read_csv(GENOTYPIC_DISTANCE_DIR / f"vf_simpson_distance_{pg_lower}.csv",index_col=0)

    # --- Subset genomes by phylogroup for dedup step ---
    genomes    = phylogroup_df.index[phylogroup_df['Phylogroup'] == pg]
    amr_bin_pg = amr_df.loc[amr_df.index.isin(genomes)]
    vf_bin_pg  = vf_df.loc[vf_df.index.isin(genomes)]

    # --- Compute ECDF input via repeated sampling ---
    ecdf_input = compute_ecdf_iterations(
        snp_sq=snp_sq, amr_jac=amr_jac, amr_sim=amr_sim,
        vf_jac=vf_jac, vf_sim=vf_sim,
        amr_bin=amr_bin_pg, vf_bin=vf_bin_pg,
        iterations=1000, percentile=0.1,
        progress_desc=f"ECDF {pg}"
    )

    # --- Prepare one figure per phylogroup ---
    plt.figure(figsize=(11.1, 8.28))
    means = {}

    # Plot all four metrics
    means['JaccardAMR'] = plot_ecdf_with_ci_fixed_x(ecdf_input['AMR Jaccard'], label='AMR Jaccard',  color=LABEL_COLOR['JaccardAMR'])
    means['SimpsonAMR'] = plot_ecdf_with_ci_fixed_x(ecdf_input['AMR Simpson'], label='AMR Simpson',  color=LABEL_COLOR['SimpsonAMR'])
    means['JaccardVF']  = plot_ecdf_with_ci_fixed_x(ecdf_input['VF Jaccard'],  label='VF Jaccard',   color=LABEL_COLOR['JaccardVF'])
    means['SimpsonVF']  = plot_ecdf_with_ci_fixed_x(ecdf_input['VF Simpson'],  label='VF Simpson',   color=LABEL_COLOR['SimpsonVF'])

    # Ax cosmetics
    plt.xlim(-0.003, 1)
    plt.ylim(0, 1.01)
    
    plt.xticks(np.linspace(0, 1, 6), fontsize=18)
    plt.yticks(np.linspace(0, 1, 6), fontsize=18)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for side in ['bottom', 'left']:
        ax.spines[side].set_linewidth(1.0)
        ax.spines[side].set_color("black")
    
    ax.grid(True, which='major', color="lightgray", linewidth=0.15, alpha=0.4)
    ax.tick_params(axis='both',which='major',
                   pad=2.8,
                   width=1,length=6)

    # Save figure
    out_png = FIGURE_OUTPUT_DIR/ f"ecdf5_{pg_lower}.pdf"
    plt.savefig(out_png, dpi=600, bbox_inches='tight',pad_inches=0.03)
    plt.close()

    # --- Manuscript numbers: zero proportion & 90th-percentile (for each metric) ---
    # Concatenate all iterations for each metric before stats
    for label_key, series_list in [
        ('AMR Jaccard',  ecdf_input['AMR Jaccard']),
        ('AMR Simpson',  ecdf_input['AMR Simpson']),
        ('VF Jaccard',   ecdf_input['VF Jaccard']),
        ('VF Simpson',   ecdf_input['VF Simpson']),
    ]:
        if len(series_list) == 0 or all(len(s) == 0 for s in series_list):
            zero_pct = np.nan
            p90      = np.nan
        else:
            all_d = np.concatenate([np.asarray(s, dtype=float) for s in series_list if len(s) > 0])
            zero_pct = float((all_d == 0).mean() * 100.0)
            p90      = float(np.percentile(all_d, 90))

        summary_rows.append({
            "Phylogroup": pg,
            "Metric": label_key,                 # e.g., "VF Simpson"
            "Zero_pct": round(zero_pct, 1),      # % of pairs with distance == 0
            "P90": round(p90, 3)                 # 90th percentile distance
        })


legend_elements = [
    Line2D([0], [0], color=LABEL_COLOR['JaccardAMR'], lw=5, label="AMR Jaccard"),
    Line2D([0], [0], color=LABEL_COLOR['SimpsonAMR'], lw=5, label="AMR Simpson"),
    Line2D([0], [0], color=LABEL_COLOR['JaccardVF'],  lw=5, label="VF Jaccard"),
    Line2D([0], [0], color=LABEL_COLOR['SimpsonVF'],  lw=5, label="VF Simpson"),
]

# Place once, e.g., bottom center of the page
plt.figure(figsize=(6, 0.28))
plt.axis("off")
plt.legend(handles=legend_elements, loc="center", ncol=4, frameon=False)
plt.savefig(FIGURE_OUTPUT_DIR / "ecdf_legend_5.pdf", dpi=600, bbox_inches="tight", pad_inches=0.01)


# ----------------------------
# Summarize
# ----------------------------
summary_df = pd.DataFrame(summary_rows)
summary_df = summary_df[['Phylogroup', 'Metric', 'Zero_pct', 'P90']]

# Also print a compact view (useful for manuscript text)
print("\n=== Summary (Zero % and 90th percentile) ===")
print(summary_df.pivot(index="Phylogroup", columns="Metric", values=["Zero_pct", "P90"]))
print(f"ECDF figures: {FIGURE_OUTPUT_DIR}/ECDF_*.pdf")

