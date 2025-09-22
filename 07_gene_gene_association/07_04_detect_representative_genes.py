from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import networkx as nx
from itertools import combinations

# ----------------------------
# Configuration
# ----------------------------
ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR_1 = ROOT / "output" / "01_preparation"
INPUT_DIR_2 = ROOT / "output" / "02_gene_screening" / "amr_and_vf_genes"
INPUT_DIR_3 = ROOT / "output" / "07_gene_gene_association" / "gene_gene_distance"
INPUT_DIR_4 = ROOT / "output" / "07_gene_gene_association" / "co_occurring_gene_clusters" 

OUTPUT_DIR = ROOT / "output" / "07_gene_gene_association" / "co_occurring_gene_clusters" 
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PHYLOGROUP_CSV = INPUT_DIR_1 / "ecoli_genomes_filtered_25080_phylogroup.csv"
AMR_CSV        = INPUT_DIR_2 / "amr_genes_presence_absence.csv"
VF_CSV         = INPUT_DIR_2 / "vf_genes_presence_absence.csv"
AMR_JACCARD_CSV = INPUT_DIR_3 / "gene_gene_distance_amr_jaccard.csv"
AMR_SIMPSON_CSV = INPUT_DIR_3 / "gene_gene_distance_amr_simpson.csv"
VF_JACCARD_CSV  = INPUT_DIR_3 / "gene_gene_distance_vf_jaccard.csv"
VF_SIMPSON_CSV  = INPUT_DIR_3 / "gene_gene_distance_vf_simpson.csv"

AMR_CLUSTER_PKL = INPUT_DIR_4 / "amr_clusters.pkl"
VF_CLUSTER_PKL  = INPUT_DIR_4 / "vf_clusters.pkl"

# ----------------------------
# Load data
# ----------------------------
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

# Sort genes alphabetically inside each clique for stable ordering
for key in amr_clusters:
    amr_clusters[key] = [sorted(gene_list) for gene_list in amr_clusters[key]]
for key in vf_clusters:
    vf_clusters[key] = [sorted(gene_list) for gene_list in vf_clusters[key]]

# ----------------------------
# Helpers
# ----------------------------
def filter_lists(nested_list, target_list):
    """Return sublists from `nested_list` that contain at least one element of `target_list`."""
    return [sublist for sublist in nested_list if any(item in sublist for item in target_list)]

# Flatten all cliques across phylogroups (used to decide coverage within a connected subgraph)
all_amr_clusters = [clq for cliques in amr_clusters.values() for clq in cliques]
all_vf_clusters  = [clq for cliques in vf_clusters.values()  for clq in cliques]

# ----------------------------
# Core: minimal representative gene set per connected subgraph
# ----------------------------
def find_minimal_gene_sets(all_gene_cluster, jaccard_data, simpson_data, gene_data, cluster_dict, subgraph):
    """
    For a given connected subgraph (set of genes), find the smallest-size gene combination(s)
    that cover all target cliques (from `all_gene_cluster`) intersecting this subgraph.
    Among combinations of the same size r, select the best one by:
      (1) minimizing the mean Jaccard distance,
      (2) if tied, minimizing the mean Simpson distance,
      (3) if still tied, maximizing the total gene-occurrence counts across relevant phylogroups,
      (4) if still tied, choosing the lexicographically smallest combination (deterministic).

    Parameters
    ----------
    all_gene_cluster : list[list[str]]
        All cliques (gene lists) relevant to the entire dataset. This is filtered
        to those that intersect the current subgraph to form the coverage target.
    jaccard_data : pd.DataFrame
        Long-format table with pairwise Jaccard distances. Must contain
        columns ['Gene 1', 'Gene 2', 'Jaccard distance', 'Phylogroup'].
    simpson_data : pd.DataFrame
        Long-format table with pairwise Simpson distances. Must contain
        columns ['Gene 1', 'Gene 2', 'Simpson distance', 'Phylogroup'].
    gene_data : pd.DataFrame
        Presence/absence matrix (rows = genomes, columns = genes) used to compute
        gene-occurrence counts in the tie-breaker.
    cluster_dict : dict[str, list[list[str]]]
        Mapping from phylogroup -> list of cliques (each clique is a list of genes)
        for that phylogroup.
    subgraph : Collection[str]
        The set of genes (nodes) in the current connected component.

    Returns
    -------
    list[str]
        The best gene combination (as a list of gene names) for this subgraph,
        selected by the priority rules described above.
    """
    # Ensure deterministic enumeration of combinations
    nodes = sorted(subgraph)

    # Cliques to be covered in this subgraph (must intersect nodes)
    target_cliques = filter_lists(all_gene_cluster, nodes)

    for r in range(1, len(nodes) + 1):
        scores = {
            "Jaccard distance": {},
            "Simpson distance": {},
            "Gene occurrence counts": {},  # per-combination list of counts used for tie-breaking
        }

        for gene_combinations in combinations(nodes, r):
            # jac_avgs_all_pgs: list of lists. For each phylogroup considered, we store
            # the per-clique averaged Jaccard distances computed under this combination.
            jac_avgs_all_pgs = []
            # simp_avgs_all_pgs: same structure for Simpson distances.
            simp_avgs_all_pgs = []

            # Coverage check: every target clique must contain at least one gene from this combination
            if all(any(g in clique for g in gene_combinations) for clique in target_cliques):
                genes_in_combo = set(gene_combinations)

                # Relevant phylogroups: those that have at least one clique intersecting this combination
                possessing_phylogroups = [
                    pg for pg, pg_cliques in cluster_dict.items()
                    if any(genes_in_combo & set(clq) for clq in pg_cliques)
                ]

                # Accumulate per-phylogroup average distances
                for phylogroup in possessing_phylogroups:
                    # jac_avgs_this_pg: per-clique averaged Jaccard distances within this phylogroup
                    jac_avgs_this_pg = []
                    # simp_avgs_this_pg: per-clique averaged Simpson distances within this phylogroup
                    simp_avgs_this_pg = []

                    pg_cliques = cluster_dict[phylogroup]
                    jac_pg = jaccard_data[jaccard_data["Phylogroup"] == phylogroup]
                    simp_pg = simpson_data[simpson_data["Phylogroup"] == phylogroup]

                    # Consider only cliques relevant to this subgraph
                    cliques = [clq for clq in pg_cliques if clq in target_cliques]
                    if cliques:
                        for clique in cliques:
                            # clique_covering_genes: genes from the combination that are present in this clique
                            clique_covering_genes = list(set(clique) & set(gene_combinations))

                            if len(clique_covering_genes) >= 2:
                                # Two or more genes from the combination appear in this clique.
                                # For each such gene, average its distance to the other genes in the clique;
                                # then take the minimum of these gene-wise averages to represent this clique.
                                jac_by_gene, simp_by_gene = {}, {}
                                for gene in clique_covering_genes:
                                    other_genes = [g for g in clique if g != gene]

                                    jd1 = jac_pg[(jac_pg["Gene 1"].isin([gene])) & (jac_pg["Gene 2"].isin(other_genes))]
                                    jd2 = jac_pg[(jac_pg["Gene 2"].isin([gene])) & (jac_pg["Gene 1"].isin(other_genes))]
                                    jaccard_dist = pd.concat([jd1, jd2])
                                    jac_edge_number = len(jaccard_dist)
                                    jac_dist = jaccard_dist["Jaccard distance"].sum()

                                    sd1 = simp_pg[(simp_pg["Gene 1"].isin([gene])) & (simp_pg["Gene 2"].isin(other_genes))]
                                    sd2 = simp_pg[(simp_pg["Gene 2"].isin([gene])) & (simp_pg["Gene 1"].isin(other_genes))]
                                    simpson_dist = pd.concat([sd1, sd2])
                                    simp_edge_number = len(simpson_dist)
                                    simp_dist = simpson_dist["Simpson distance"].sum()

                                    jac_by_gene[gene]  = jac_dist / jac_edge_number
                                    simp_by_gene[gene] = simp_dist / simp_edge_number

                                # clique_jac_avg: representative Jaccard distance for this clique in this phylogroup
                                clique_jac_avg = min(jac_by_gene.values())
                                # clique_simp_avg: representative Simpson distance for this clique in this phylogroup
                                clique_simp_avg = min(simp_by_gene.values())

                            else:
                                # Exactly one gene from the combination is present in this clique.
                                # Average distances from that single gene to all other genes in the clique.
                                other_genes = [g for g in clique if g not in clique_covering_genes]

                                jaccard_dist = pd.concat([
                                    jac_pg[(jac_pg["Gene 1"].isin(clique_covering_genes)) & (jac_pg["Gene 2"].isin(other_genes))],
                                    jac_pg[(jac_pg["Gene 2"].isin(clique_covering_genes)) & (jac_pg["Gene 1"].isin(other_genes))],
                                ])
                                jac_edge_number = len(jaccard_dist)
                                jac_dist = jaccard_dist["Jaccard distance"].sum()

                                simpson_dist = pd.concat([
                                    simp_pg[(simp_pg["Gene 1"].isin(clique_covering_genes)) & (simp_pg["Gene 2"].isin(other_genes))],
                                    simp_pg[(simp_pg["Gene 2"].isin(clique_covering_genes)) & (simp_pg["Gene 1"].isin(other_genes))],
                                ])
                                simp_edge_number = len(simpson_dist)
                                simp_dist = simpson_dist["Simpson distance"].sum()

                                # clique_jac_avg / clique_simp_avg: representative distances for this clique in this phylogroup
                                clique_jac_avg  = jac_dist / jac_edge_number
                                clique_simp_avg = simp_dist / simp_edge_number

                            jac_avgs_this_pg.append(clique_jac_avg)
                            simp_avgs_this_pg.append(clique_simp_avg)

                    # Accumulate per-phylogroup lists
                    jac_avgs_all_pgs.append(jac_avgs_this_pg)
                    simp_avgs_all_pgs.append(simp_avgs_this_pg)

                # Compute global means across all phylogroups and cliques (flatten first)
                flat_jac  = [v for per_pg in jac_avgs_all_pgs  for v in per_pg]
                flat_simp = [v for per_pg in simp_avgs_all_pgs for v in per_pg]
                if not flat_jac or not flat_simp:
                    continue

                mean_jac_distance  = float(np.mean(flat_jac))
                mean_simp_distance = float(np.mean(flat_simp))

                # Tie-breaker: for each gene in this candidate combination and for each relevant
                # phylogroup, count how many genomes (rows) carry that gene in the presence/
                # absence matrix (gene_data filtered by phylogroup_df). These per-gene,
                # per-phylogroup counts are later summed to prefer combinations with stronger support.
                gene_occurrence_counts = []
                for idx in range(r):
                    for phylogroup in possessing_phylogroups:
                        pg_idx = phylogroup_df.index[phylogroup_df["Phylogroup"] == phylogroup]
                        pg_gene_matrix = gene_data[gene_data.index.isin(pg_idx)]
                        presence_by_gene = pg_gene_matrix.sum().to_dict()
                        gene_occurrence_counts.append(presence_by_gene.get(gene_combinations[idx], 0))

                scores["Jaccard distance"][gene_combinations] = mean_jac_distance
                scores["Simpson distance"][gene_combinations] = mean_simp_distance
                scores["Gene occurrence counts"][gene_combinations] = gene_occurrence_counts

        # Select best combination for size r:
        # (1) min Jaccard → (2) min Simpson among ties → (3) max summed occurrences → (4) lexicographically smallest
        if scores["Jaccard distance"]:
            min_jac = min(scores["Jaccard distance"].values())
            jac_keys = [k for k, v in scores["Jaccard distance"].items() if v == min_jac]

            if len(jac_keys) == 1:
                best_gene_combinations = list(jac_keys[0])
            else:
                min_sim = min(scores["Simpson distance"][k] for k in jac_keys)
                sim_keys = [k for k in jac_keys if scores["Simpson distance"][k] == min_sim]

                if len(sim_keys) == 1:
                    best_gene_combinations = list(sim_keys[0])
                else:
                    max_support = max(sum(scores["Gene occurrence counts"][k]) for k in sim_keys)
                    top_keys = [k for k in sim_keys if sum(scores["Gene occurrence counts"][k]) == max_support]
                    # Final deterministic tie-break: pick lexicographically smallest tuple
                    best_gene_combinations = list(min(top_keys))

            break  # Found the best combination for the smallest size r
        else:
            continue

    return best_gene_combinations

# ----------------------------
# Build co-occurrence networks and find minimal representatives
# ----------------------------
def create_network(groups):
    """Build an undirected graph by connecting genes that co-occur in the same clique."""
    G = nx.Graph()
    for group in groups:
        for gene1, gene2 in combinations(group, 2):
            G.add_edge(gene1, gene2)
    return list(nx.connected_components(G))

# AMR
connected_components_amr = create_network(all_amr_clusters)
all_minimal_sets_amr = []
for component in connected_components_amr:
    minimal = find_minimal_gene_sets(
        all_gene_cluster=all_amr_clusters,
        jaccard_data=amr_jaccard,
        simpson_data=amr_simpson,
        gene_data=amr_df,
        cluster_dict=amr_clusters,
        subgraph=component,
    )
    all_minimal_sets_amr.extend(minimal)
all_minimal_sets_amr = sorted(all_minimal_sets_amr)

# VF
connected_components_vf = create_network(all_vf_clusters)
all_minimal_sets_vf = []
for component in connected_components_vf:
    minimal = find_minimal_gene_sets(
        all_gene_cluster=all_vf_clusters,
        jaccard_data=vf_jaccard,
        simpson_data=vf_simpson,
        gene_data=vf_df,
        cluster_dict=vf_clusters,
        subgraph=component,
    )
    all_minimal_sets_vf.extend(minimal)
all_minimal_sets_vf = sorted(all_minimal_sets_vf)

# ----------------------------
# Save outputs
# ----------------------------
with open(OUTPUT_DIR / "represent_gene_amr.pkl", "wb") as f:
    pickle.dump(all_minimal_sets_amr, f)
with open(OUTPUT_DIR / "represent_gene_vf.pkl", "wb") as f:
    pickle.dump(all_minimal_sets_vf, f)

