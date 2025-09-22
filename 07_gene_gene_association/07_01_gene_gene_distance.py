import pandas as pd
import numpy as np
import os
import itertools
from sklearn.metrics import jaccard_score
import py_stringmatching as sm
from statsmodels.stats.contingency_tables import Table2x2
from pathlib import Path

# ----------------------------
# Configuration 
# ----------------------------
ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = ROOT / "output"
OUTPUT_DIR = ROOT / "output" / "07_gene_gene_association" /"gene_gene_distance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PHYLOGROUP_CSV = INPUT_DIR / "01_preparation" /"ecoli_genomes_filtered_25080_phylogroup.csv"
ST_CSV = INPUT_DIR / "01_preparation" / "ecoli_genomes_filtered_25080_MLST.csv"
AMR_CSV = INPUT_DIR / "02_gene_screening" / "amr_and_vf_genes" / "amr_genes_presence_absence.csv"
VF_CSV = INPUT_DIR / "02_gene_screening" / "amr_and_vf_genes" / "vf_genes_presence_absence.csv"

phylogroup_df = pd.read_csv(PHYLOGROUP_CSV, index_col=0)
st_df = pd.read_csv(ST_CSV, index_col=0)
amr_df = pd.read_csv(AMR_CSV, index_col=0)
vf_df = pd.read_csv(VF_CSV, index_col=0)

# Define operon gene lists
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

# Flatten and filter VF genes
operons=[afa,ces,chu,ent,esc,esp,fep,fim,gsp,hly,iro,irp,iuc,kps,nle,pap,sep,sfa,shu,stx,yag,ybt]
flattend_operons=list(itertools.chain.from_iterable(operons))
vf_df=vf_df[flattend_operons]

# Calculate the number of evaluated gene pairs
phylogroups = ['A','B1','B2','C','D','E','F','G']

# Filter genes detected in at least 20% of genomes
def over20_percent(df):
    threshold = len(df) * 0.2
    cols = df.columns[df.sum(axis=0) >= threshold]
    return df[cols]

#AMR gene number
for phylogroup, idx in phylogroup_df.groupby('Phylogroup').groups.items():
    amr_ph = amr_df.loc[idx]
    count = len(over20_percent(amr_ph).columns)
    print(f"AMR gene number of Phylogroup {phylogroup}: {count}")

#VF gene number
for phylogroup, idx in phylogroup_df.groupby('Phylogroup').groups.items():
    vf_ph = vf_df.loc[idx]
    count = len(over20_percent(vf_ph).columns)
    print(f"VF gene number of Phylogroup {phylogroup}: {count}")

jac = sm.Jaccard()
oc = sm.OverlapCoefficient()

# Jaccard and Simpson calculations
def jaccard_calculate(df, gene_pairs):
    results = []
    for g1, g2 in gene_pairs:
        d1, d2 = df[g1], df[g2]
        jaccard_dist = 1 - jaccard_score(d1, d2)

        contingency = pd.crosstab(d1, d2).reindex(index=[0,1], columns=[0,1], fill_value=0)

        table_int = np.round(contingency).astype(int)
        fisher = Table2x2(table_int)
        odds_ratio = fisher.oddsratio
        p_value = fisher.test_nominal_association().pvalue

        n1, n2 = d1.sum(), d2.sum()
        gene1, gene2 = sorted([(n1, g1), (n2, g2)])

        results.append([gene1[1], gene2[1], jaccard_dist, odds_ratio, p_value, gene1[0], gene2[0]])
    return pd.DataFrame(results, columns=['Gene 1','Gene 2','Jaccard distance','Odds ratio','P-value','No. of gene 1','No. of gene 2'])

def simpson_calculate(df, gene_pairs):
    overlap = sm.OverlapCoefficient()
    results = []
    for g1, g2 in gene_pairs:
        d1, d2 = df[g1], df[g2]
        contingency = pd.crosstab(d1, d2).reindex(index=[0,1], columns=[0,1], fill_value=0)

        table_int = np.round(contingency).astype(int)
        fisher = Table2x2(table_int)
        odds_ratio = fisher.oddsratio
        p_value = fisher.test_nominal_association().pvalue

        pos_idx1 = d1[d1==1].index.tolist()
        pos_idx2 = d2[d2==1].index.tolist()
        simpson_sim = overlap.get_sim_score(pos_idx1, pos_idx2)
        simpson_dist = 1 - simpson_sim

        n1, n2 = len(pos_idx1), len(pos_idx2)
        gene1, gene2 = sorted([(n1, g1), (n2, g2)])

        results.append([gene1[1], gene2[1], simpson_dist, odds_ratio, p_value, gene1[0], gene2[0]])
    return pd.DataFrame(results, columns=['Gene 1','Gene 2','Simpson distance','Odds ratio','P-value','No. of gene 1','No. of gene 2'])

# Start main processing
amr_jaccard_all, amr_simpson_all = [], []
vf_jaccard_all, vf_simpson_all = [], []

for phylogroup in phylogroups:
    indices = phylogroup_df[phylogroup_df['Phylogroup'] == phylogroup].index
    
    # AMR data processing
    amr_sub = amr_df.loc[indices]
    amr_sub = over20_percent(amr_sub)
    amr_genes = list(amr_sub.columns)
    amr_combinations = list(itertools.combinations(amr_genes, 2))

    if amr_combinations:  # Execute only if gene pairs exist
        amr_j = jaccard_calculate(amr_sub, amr_combinations)
        amr_s = simpson_calculate(amr_sub, amr_combinations)
        amr_j['Phylogroup'] = phylogroup
        amr_s['Phylogroup'] = phylogroup
        amr_jaccard_all.append(amr_j)
        amr_simpson_all.append(amr_s)

    # VF data processing
    vf_sub = vf_df.loc[indices]
    vf_sub = over20_percent(vf_sub)
    vf_genes = list(vf_sub.columns)
    vf_combinations = list(itertools.combinations(vf_genes, 2))

    if vf_combinations:
        vf_j = jaccard_calculate(vf_sub, vf_combinations)
        vf_s = simpson_calculate(vf_sub, vf_combinations)
        vf_j['Phylogroup'] = phylogroup
        vf_s['Phylogroup'] = phylogroup
        vf_jaccard_all.append(vf_j)
        vf_simpson_all.append(vf_s)

# Merge results
amr_jaccard = pd.concat(amr_jaccard_all)
amr_simpson = pd.concat(amr_simpson_all)

vf_jaccard  = pd.concat(vf_jaccard_all)
vf_simpson  = pd.concat(vf_simpson_all)

# Save output files
amr_jaccard_output = OUTPUT_DIR / "gene_gene_distance_amr_jaccard.csv"
amr_simpson_output = OUTPUT_DIR / "gene_gene_distance_amr_simpson.csv"
vf_jaccard_output = OUTPUT_DIR / "gene_gene_distance_vf_jaccard.csv"
vf_simpson_output = OUTPUT_DIR / "gene_gene_distance_vf_simpson.csv"

amr_jaccard.to_csv(amr_jaccard_output)
amr_simpson.to_csv(amr_simpson_output)
vf_jaccard.to_csv(vf_jaccard_output)
vf_simpson.to_csv(vf_simpson_output)
