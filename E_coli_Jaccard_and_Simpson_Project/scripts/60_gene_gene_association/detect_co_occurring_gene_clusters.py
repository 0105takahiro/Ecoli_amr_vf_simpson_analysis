from pathlib import Path
import random
import pickle

import pandas as pd
import networkx as nx
from docplex.mp.model import Model

# ----------------------------
# Configuration 
# ----------------------------
ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR_1 = ROOT / "Output"/ "Preparation"
INPUT_DIR_2 = ROOT / "Output"/ "AMR_and_VF_genes_ok"
INPUT_DIR_3 = ROOT / "Output"/ "gene_gene_distance"

OUTPUT_DIR = ROOT / "Output" / "co_occurring_gene_clusters"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PHYLOGROUP_CSV = INPUT_DIR_1 /"ecoli-genomes_filtered_25080_phylogroup.csv"
AMR_CSV = INPUT_DIR_2 / "amr-genes-presence-absence.csv"
VF_CSV = INPUT_DIR_2  / "vf-genes-presence-absence.csv"
AMR_JACCARD_CSV = INPUT_DIR_3 / "gene_gene_distance_amr_jaccard.csv"
AMR_SIMPSON_CSV = INPUT_DIR_3 / "gene_gene_distance_amr_simpson.csv"
VF_JACCARD_CSV  = INPUT_DIR_3 / "gene_gene_distance_vf_jaccard.csv"
VF_SIMPSON_CSV  = INPUT_DIR_3 / "gene_gene_distance_vf_simpson.csv"

phylogroup_df =pd.read_csv(PHYLOGROUP_CSV, index_col=0)
amr_df =pd.read_csv(AMR_CSV,index_col=0)
vf_df  =pd.read_csv(VF_CSV,index_col=0)
amr_jaccard = pd.read_csv(AMR_JACCARD_CSV, index_col=0)
amr_simpson = pd.read_csv(AMR_SIMPSON_CSV, index_col=0)
vf_jaccard  = pd.read_csv(VF_JACCARD_CSV, index_col=0)
vf_simpson  = pd.read_csv(VF_SIMPSON_CSV, index_col=0)

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

def extract_within_group_distances(data, groups):
    """
    Extract pairwise distances within predefined gene groups and return them as a sorted list.

    Parameters
    ----------
    data : pd.DataFrame
        Distance matrix in long format. Must contain columns ['Gene 1', 'Gene 2', 'Jaccard distance'].
    groups : list[list[str]]
        List of gene groups (each group is a list of gene names).

    Returns
    -------
    list[float]
        A list of Jaccard distances for gene pairs found within the same group,
        sorted in descending order.
    """
    within_group_df = pd.DataFrame()
    for group in groups:
        group_set = set(group)  # Convert group into a set of nodes
        filtered_df = data[data["Gene 1"].isin(group_set) & data["Gene 2"].isin(group_set)]
        if not filtered_df.empty:
            within_group_df = pd.concat([within_group_df, filtered_df])

    within_group_df = within_group_df.sort_values(by='Jaccard distance', ascending=False)
    return list(within_group_df['Jaccard distance'])

def select_min_distance_list(new, current):
    """
    Compare two lists element-wise and return the lexicographically smaller one.

    Parameters
    ----------
    new : list[float]
        Candidate list of distances (e.g., Jaccard or Simpson distances).
    current : list[float]
        Current best list of distances for comparison.

    Returns
    -------
    list[float]
        The list that is lexicographically smaller.
        - If `new` < `current` at the first differing element, return `new`.
        - If `new` > `current` at the first differing element, return `current`.
        - If all compared elements up to the shorter length are equal,
        the shorter list is returned.
        - If the two lists are completely identical, return `new`.
    """
    if new != current:
        for x, y in zip(new, current):
            if x < y:
                return new
            elif x > y:
                return current
        return new if len(new) < len(current) else current
    else:
        return new

def random_clique_covers_multiple_runs(edges, K, num_runs, num_solutions):
    """
    Generate multiple clique cover solutions by partitioning the graph into K cliques,
    using randomized modifications across multiple runs.

    Parameters
    ----------
    edges : list[tuple]
        List of edges defining the graph (pairs of nodes).
    K : int
        Number of cliques to cover the graph.
    num_runs : int
        Number of independent runs to perform. Randomness is introduced in each run
        to explore different feasible solutions.
    num_solutions : int
        Maximum number of unique clique cover solutions to collect per run.

    Returns
    -------
    list[list[list[str]]]
        A list of clique cover solutions. Each solution is represented as a list of cliques,
        and each clique is represented as a list of nodes. Singleton nodes are removed
        (only cliques with size > 1 are retained).

    Notes
    -----
    - Each node must belong to exactly one clique.
    - Non-adjacent nodes are constrained from being placed in the same clique.
    - The optimization model is formulated with binary decision variables, solved
      using `docplex`.
    - To introduce variability across solutions, a random subset of edges is removed
      iteratively between solves.
    - Solutions are collected only if they are unique across runs.
    - Internally, singleton cliques (size == 1) are allowed. However, they are removed 
    　in the final output because they do not contribute to the distance calculation 
    　in the subsequent step.
    """
    all_solutions = []

    for _ in range(num_runs):
        G = nx.Graph()
        G.add_edges_from(edges)

        nodes = list(G.nodes)
        edges = list(G.edges)
        n = len(nodes)

        m = Model('Clique_Cover')

        # Create variables
        x = {(i, k): m.binary_var(name=f'x_{i}_{k}') for i in range(n) for k in range(K)}

        # Constraint 1: Each node must be in at least one clique
        for i in range(n):
            m.add_constraint(m.sum(x[i, k] for k in range(K)) == 1)

        # Constraint 2: Non-adjacent nodes cannot be in the same clique
        for k in range(K):
            for i in range(n):
                for j in range(i + 1, n):
                    if not G.has_edge(nodes[i], nodes[j]):
                        m.add_constraint(x[i, k] + x[j, k] <= 1)

        # Objective (minimize the number of cliques)
        m.minimize(m.sum(x[i, k] for i in range(n) for k in range(K)))

        solutions = []
        while len(solutions) < num_solutions:
            m.solve()
            if m.solution is None:
                break

            # Extract solution
            clusters = [[] for _ in range(K)]
            for i in range(n):
                for k in range(K):
                    if x[i, k].solution_value > 0.5:
                        clusters[k].append(nodes[i])

            # Check for uniqueness and add new solutions
            if clusters not in solutions:
                solutions.append(clusters)

            # Add randomness by cutting some edges
            remove_edges = random.sample(edges, k=min(len(edges) // 2, 1))
            G.remove_edges_from(remove_edges)

            # Update model
            m.clear_constraints()
            for i in range(n):
                m.add_constraint(m.sum(x[i, k] for k in range(K)) == 1)
            for k in range(K):
                for i in range(n):
                    for j in range(i + 1, n):
                        if not G.has_edge(nodes[i], nodes[j]):
                            m.add_constraint(x[i, k] + x[j, k] <= 1)

        # After solutions are found, add them to all_solutions if they are unique
        for solution in solutions:
            if solution not in all_solutions:
                all_solutions.append(solution)
    all_clique_forming_solution = [[list_ for list_ in middle if len(list_) > 1] for middle in all_solutions]
    return all_clique_forming_solution



def find_optimal_clique_cover(data, edges, K, num_runs, num_solutions):
    """
    Select the best clique cover (among randomized candidates) that minimizes
    intra-clique distances in a lexicographic sense.

    Summary
    -------
    This function generates multiple candidate clique covers with
    `random_clique_covers_multiple_runs` and picks the one whose list of
    intra-clique distances (computed from `data`) is lexicographically
    smallest. Distances are extracted by `extract_within_group_distances`,
    which sorts the distances in descending order and the comparison uses
    `select_min_distance_list` (lexicographic minimization).

    Parameters
    ----------
    data : pd.DataFrame
        Long-format pairwise distance table. It must contain at least the
        following columns:
            - 'Gene 1' : str
            - 'Gene 2' : str
            - 'Jaccard distance' : float
    edges : list[tuple[str, str]]
        Edge list defining the graph (pairs of genes).
    K : int
        Number of cliques to use in each candidate cover.
    num_runs : int
        Number of independent randomized runs to explore different solutions.
    num_solutions : int
        Maximum number of unique solutions to collect per run.

    Returns
    -------
    list[list[str]]
        The selected clique cover as a list of cliques (each clique is a list of
        gene names). Singleton cliques (size == 1) are removed by
        `random_clique_covers_multiple_runs` prior to evaluation and return.

    Notes
    -----
    - Candidate solutions are produced by `random_clique_covers_multiple_runs`,
      which internally may allow singleton cliques but removes them before
      returning the final candidate list.
    - The distance list for each candidate is obtained by
      `extract_within_group_distances(data, candidate)` and compared using
      `select_min_distance_list(new, current)` to perform lexicographic
      minimization (preferring smaller distances; ties broken by shorter lists).
    """
    # Initial candidate and its distance profile
    current_best_solution = random_clique_covers_multiple_runs(edges, K, num_runs, num_solutions)[0]
    current_min_dist = extract_within_group_distances(data, current_best_solution)
    best_candidate = (current_best_solution, current_min_dist)

    # Evaluate all candidates
    candidates = random_clique_covers_multiple_runs(edges, K, num_runs, num_solutions)
    for i, new_solution in enumerate(candidates):
        # Keep a stable representation if needed elsewhere
        # new_solution_tuple = tuple(sorted(map(tuple, new_solution)))
        new_dist = extract_within_group_distances(data, new_solution)

        if select_min_distance_list(new_dist, current_min_dist) == new_dist:
            current_best_solution = new_solution
            current_min_dist = new_dist
            best_candidate = (current_best_solution, current_min_dist)

    best_candidate = best_candidate[0]
    best_cluster_forming_candidate = [lst for lst in best_candidate if len(lst) > 1]
    return best_cluster_forming_candidate


def determine_amr_clique(data, num_runs, num_solutions):
    """
    Build co-occurring AMR gene cliques for each phylogroup.

    Workflow
    --------
    For each phylogroup:
      1) Filter the pairwise edges to this phylogroup and to sufficiently co-occurring
         pairs (here: Jaccard distance <= 0.2). These filtered pairs form the edges
         of an undirected graph whose nodes are genes.
      2) Split the graph into connected components.
      3) For each connected component:
         - If the component has exactly two nodes, treat that pair as a clique.
         - Otherwise, try to cover the component with K cliques (K = 1..29):
             * First, generate randomized clique-cover candidates with
               `random_clique_covers_multiple_runs(...)`. If exactly one candidate
               solution exists, accept it.
             * If multiple candidate solutions exist, choose the best one via
               `find_optimal_clique_cover(...)`, which compares intra-clique
               distance profiles lexicographically.
      4) Collect all cliques discovered for this phylogroup.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format AMR Jaccard table (must include columns: 'Gene 1', 'Gene 2',
        'Jaccard distance', 'Phylogroup').
    num_runs : int
        Number of randomized runs to propose multiple candidate clique covers.
    num_solutions : int
        Maximum number of unique candidate solutions to collect per run.

    Returns
    -------
    dict[str, list[list[str]]]
        Mapping from phylogroup -> list of cliques (each clique is a list of gene names).

    Notes
    -----
    - The Jaccard threshold (0.2) defines which gene pairs are considered "co-occurring"
      enough to add an edge; tune this threshold based on your dataset.
    - `random_clique_covers_multiple_runs` internally allows singleton cliques but removes
      them in its output; this function, too, only keeps cliques with size > 1.
    """
    cluster_dict = {}
    for phylogroup in phylogroups:
        # 1) Restrict to this phylogroup and to sufficiently close pairs
        data2 = data[data['Phylogroup'] == phylogroup]
        data2 = data2[data2['Jaccard distance'] <= 0.2]

        # Build graph and split into connected components
        edges = data2[["Gene 1", "Gene 2"]].values.tolist()
        G = nx.Graph()
        G.add_edges_from(edges)
        S = [G.subgraph(c).copy() for c in nx.connected_components(G)]

        phylogroup_cliques = []  # accumulator of cliques for this phylogroup
        for subgraph in S:
            nodes = list(subgraph.nodes())
            subgraph_edges = subgraph.edges()

            if len(nodes) == 2:
                # Exactly two nodes: treat the pair as a clique
                phylogroup_cliques.append(nodes)
            else:
                # Try increasing K until we can cover this component
                for K in range(1, len(nodes)):
                    # Quick probe for feasible covers using a capped search
                    # (You may pass through num_runs/num_solutions if you prefer.)
                    cliques_in_subgraph = random_clique_covers_multiple_runs(
                        subgraph_edges, K, num_runs=num_runs, num_solutions=num_solutions
                    )
                    if cliques_in_subgraph != []:
                        if len(cliques_in_subgraph) == 1:
                            # Only one candidate solution: accept it
                            phylogroup_cliques.extend(cliques_in_subgraph[0])
                            break
                        else:
                            # Multiple candidates: select the one minimizing intra-clique distances
                            clique_list_represent = find_optimal_clique_cover(
                                data2, subgraph_edges, K, num_runs, num_solutions
                            )
                            phylogroup_cliques.extend(clique_list_represent)
                            break
                    else:
                        continue

        cluster_dict[phylogroup] = phylogroup_cliques

    return cluster_dict

def determine_vf_clique(data, num_runs, num_solutions):
    """
    Build co-occurring VF gene cliques for each phylogroup, scoped within known operons.

    Workflow
    --------
    For each phylogroup:
      For each predefined operon (list of genes):
        1) Filter the pairwise edges to this phylogroup AND to gene pairs within
           the same operon (operon-scoped subgraph). Also require Jaccard distance <= 0.2.
        2) Build the graph for that operon and split into connected components.
        3) For each component:
           - If it has exactly two nodes, treat the pair as a clique.
           - Otherwise, try to cover the component with K cliques (K = 1..29):
               * Propose randomized clique-cover candidates with
                 `random_clique_covers_multiple_runs(...)`. If there is exactly one
                 candidate, accept it.
               * If multiple candidates exist, pick the best one via
                 `find_optimal_clique_cover(...)` (lexicographic distance minimization).
      Finally, aggregate all cliques across operons for that phylogroup.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format VF Jaccard table (must include columns: 'Gene 1', 'Gene 2',
        'Jaccard distance', 'Phylogroup').
    num_runs : int
        Number of randomized runs to propose multiple candidate clique covers.
    num_solutions : int
        Maximum number of unique candidate solutions to collect per run.

    Returns
    -------
    dict[str, list[list[str]]]
        Mapping from phylogroup -> list of cliques (each clique is a list of gene names).

    Notes
    -----
    - The search is restricted to **within-operon** pairs to reflect known operon structure.
    - The Jaccard threshold (0.2) again controls which pairs are considered as edges.
    - Singleton cliques are discarded (size > 1 only).
    """
    cluster_dict = {}
    for phylogroup in phylogroups:
        data2 = data[data['Phylogroup'] == phylogroup]
        phylogroup_cliques = []  # accumulator of cliques for this phylogroup

        for operon in operons:
            # 1) Restrict to current operon and to close pairs
            data3 = data2[(data2['Gene 1'].isin(operon)) & (data2['Gene 2'].isin(operon))]
            data3 = data3[data3['Jaccard distance'] <= 0.2]

            # Build graph and split into connected components
            edges = data3[["Gene 1", "Gene 2"]].values.tolist()
            G = nx.Graph()
            G.add_edges_from(edges)
            S = [G.subgraph(c).copy() for c in nx.connected_components(G)]

            for subgraph in S:
                nodes = list(subgraph.nodes())
                subgraph_edges = subgraph.edges()

                if len(nodes) == 2:
                    # Exactly two nodes: treat the pair as a clique
                    phylogroup_cliques.append(nodes)
                else:
                    # Try increasing K until we can cover this component
                    for K in range(1, len(nodes)):
                        cliques_in_subgraph = random_clique_covers_multiple_runs(
                            subgraph_edges, K, num_runs, num_solutions
                        )
                        if cliques_in_subgraph != []:
                            if len(cliques_in_subgraph) == 1:
                                phylogroup_cliques.extend(cliques_in_subgraph[0])
                                break
                            else:
                                clique_list_represent = find_optimal_clique_cover(
                                    data3, subgraph_edges, K, num_runs, num_solutions
                                )
                                phylogroup_cliques.extend(clique_list_represent)
                                break
                        else:
                            continue

        cluster_dict[phylogroup] = phylogroup_cliques

    return cluster_dict

# Compute co-occurring cliques for AMR/VF.
amr_clusters=determine_amr_clique(amr_jaccard,1000,1000)
vf_clusters=determine_vf_clique(vf_jaccard,1000,1000)

# ----------------------------
# Save results
# ----------------------------
with open( OUTPUT_DIR/'amr_clusters.pkl', 'wb') as f:
    pickle.dump(amr_clusters, f)
with open( OUTPUT_DIR/'vf_clusters.pkl', 'wb') as f:
    pickle.dump(vf_clusters, f)
