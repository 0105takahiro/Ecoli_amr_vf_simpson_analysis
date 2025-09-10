import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from pathlib import Path

# ----------------------------
# Configuration 
# ----------------------------
ROOT = Path(__file__).resolve().parents[2]

FIGURE_OUTPUT_DIR = ROOT/ "figures"/ "umap"
FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PHYLOGROUP_CSV = ROOT / "output" / "preparation" /"ecoli-genomes_filtered_25080_phylogroup_ok.csv"
AMR_CSV = ROOT / "output" / "amr_and_vf_genes" / "amr-genes-presence-absence_ok.csv"
AMR_DISTANCE_MATRIX_CSV = ROOT / "output" / "distance_matrix_all_phylogroups" / "amr_simpson_distance_matrix_all_phylogroups.csv"

phylogroup_df = pd.read_csv(PHYLOGROUP_CSV, index_col=0)
amr_df = pd.read_csv(AMR_CSV, index_col=0)
amr_distance_matrix =pd.read_csv(AMR_DISTANCE_MATRIX_CSV,index_col=0)

#UMAP dimensionality reduction
umap_model = umap.UMAP(
    random_state=0, n_neighbors=20, metric="precomputed",
    n_components=2, min_dist=0.25
)
embedding = umap_model.fit_transform(amr_distance_matrix)
embedding_df = pd.DataFrame(embedding, index=amr_distance_matrix.index, columns=["UMAP1", "UMAP2"])
embedding_merged = embedding_df.join(phylogroup_df).dropna()


# ----------------------------
# Scatter plot (no axes, no legend)
# ----------------------------
color_palette = ["#4670ef", "#f2c475", "#6ce187", "#f7605a",
                 "#a367f1", "#b98859", "#f89be5", "#bab8b8"]
custom_palette = sns.color_palette(color_palette)

plt.figure(figsize=(80, 60))
sns.set_context("paper")
sns.set_style("white")

ax = sns.scatterplot(
    data=embedding_merged,
    x="UMAP1", y="UMAP2",
    hue="Phylogroup",
    hue_order=["A","B1","B2","C","D","E","F","G"],
    palette=custom_palette,
    s=200, linewidth=0.1, alpha=1
)

# Calculate axis limits
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
xlength = xmax - xmin
ylength = ymax - ymin

# Draw boundary lines with margins (these were manually determined based on UMAP projection)
ax.vlines(x=7.7, ymin=ymin-ylength*0.01, ymax=ymax+ylength*0.01, ls="--", lw=5, colors="grey")
ax.vlines(x=14 , ymin=ymin-ylength*0.01, ymax=ymax+ylength*0.01, ls="--", lw=5, colors="grey")
ax.hlines(y=15.8, xmin=14, xmax=xmax+xlength*0.01, ls="--", lw=5, colors="grey")

# Remove axes and pads
ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
ax.set_xlim(xmin-xlength*0.01, xmax+xlength*0.01)
ax.set_ylim(ymin-ylength*0.01, ymax+ylength*0.01)
ax.get_legend().remove()
plt.tight_layout()
plt.savefig(FIGURE_OUTPUT_DIR, dpi=600, bbox_inches="tight")

#Clustering based on boundaries
cluster_conditions = [
    (embedding_merged["UMAP1"] < 7.7),
    (embedding_merged["UMAP1"] >= 7.7) & (embedding_merged["UMAP1"] < 14),
    (embedding_merged["UMAP1"] >= 14) & (embedding_merged["UMAP2"] >= 15.8),
    (embedding_merged["UMAP1"] >= 14) & (embedding_merged["UMAP2"] <  15.8)
]
clusters = pd.concat([
    embedding_merged[cond].assign(cluster=i+1) 
    for i, cond in enumerate(cluster_conditions)
])

#Count phylogroup composition per cluster
composition = clusters.groupby("cluster")["Phylogroup"].value_counts().unstack().fillna(0)
composition = composition[["A","B1","B2","C","D","E","F","G"]]

# Compute specific phylogroup proportions
def cluster_fraction(df, clusters, phylogroup):
    total = df[phylogroup].sum()
    cluster_sum = df.loc[clusters, phylogroup].sum()
    return round(cluster_sum / total * 100, 1)

ratios = {
    "B1 in clusters 1+2 (%)": cluster_fraction(composition, [1, 2], "B1"),
    "C in clusters 1+2 (%)": cluster_fraction(composition, [1, 2], "C"),
    "E in clusters 1+2 (%)": cluster_fraction(composition, [1, 2], "E"),
    "A in clusters 3+4 (%)": cluster_fraction(composition, [3, 4], "A"),
    "B2 in clusters 3+4 (%)": cluster_fraction(composition, [3, 4], "B2")
}
print(ratios)

#Pie chart plotting function
def pie_plot(cluster_num: int)-> None:
    subset = clusters[clusters["cluster"] == cluster_num]
    counts = subset["Phylogroup"].value_counts().reindex(["A","B1","B2","C","D","E","F","G"]).fillna(0)
    OUT_PATH = os.path.join(FIGURE_OUTPUT_DIR, f"cluster{cluster_num}_amr_simpson_pie_chart.pdf")
    plt.figure(figsize=(15, 15))
    plt.pie(counts, counterclock=False, startangle=90, labeldistance=None, colors=custom_palette)
    plt.savefig(  FIGURE_OUTPUT_DIR / "umap_projection_amr_simpson.pdf", dpi=600, bbox_inches="tight")
    plt.close()

#%Output pie charts
for cluster_num in [1,2,3,4]:
    pie_plot(cluster_num)

#Extract binary matrix per cluster
cluster_indices = {i: clusters[clusters["cluster"]==i].index for i in [1,2,3,4]}
amr_clusters = {i: amr_df.loc[idx] for i, idx in cluster_indices.items()}

#Count combinations of glpT and pmrB mutations
for i in [1,2,3,4]:
    amr = amr_clusters[i]
    print(f"Cluster {i} glpT(+) & pmrB(+):", ((amr["glpT_E448K"]==1) & (amr["pmrB_Y358N"]==1)).sum())
    print(f"Cluster {i} glpT(+) & pmrB(-):", ((amr["glpT_E448K"]==1) & (amr["pmrB_Y358N"]==0)).sum())
    print(f"Cluster {i} glpT(-) & pmrB(+):", ((amr["glpT_E448K"]==0) & (amr["pmrB_Y358N"]==1)).sum())
    print(f"Cluster {i} glpT(-) & pmrB(-):", ((amr["glpT_E448K"]==0) & (amr["pmrB_Y358N"]==0)).sum())
    print(f"Cluster {i} total:", len(amr))

# Summary percentages for selected pairs
# Cluster 1
n_glpT1_pmrB1 = ((amr_clusters[1]["glpT_E448K"]==1) & (amr_clusters[1]["pmrB_Y358N"]==1)).sum()
n_total1 = len(amr_clusters[1])
print(f"Cluster 1: glpT(+), pmrB(+) → {n_glpT1_pmrB1}/{n_total1} = {n_glpT1_pmrB1/n_total1:.2%}")

# Cluster 2
n_glpT2_pmrB0 = ((amr_clusters[2]["glpT_E448K"]==1) & (amr_clusters[2]["pmrB_Y358N"]==1)).sum()
n_total2 = len(amr_clusters[2])
print(f"Cluster 2: glpT(+), pmrB(+) → {n_glpT2_pmrB0}/{n_total2} = {n_glpT2_pmrB0/n_total2:.2%}")

# Cluster 3
n_glpT3_pmrB3 = ((amr_clusters[3]["glpT_E448K"]==1) & (amr_clusters[3]["pmrB_Y358N"]==0)).sum()
n_total3 = len(amr_clusters[3])
print(f"Cluster 3: glpT(+), pmrB(-) → {n_glpT3_pmrB3}/{n_total3} = {n_glpT3_pmrB3/n_total3:.2%}")

# Cluster 4
n_glpT4_pmrB1 = ((amr_clusters[4]["glpT_E448K"]==0) & (amr_clusters[4]["pmrB_Y358N"]==0)).sum()
n_total4 = len(amr_clusters[4])
print(f"Cluster 4: glpT(-), pmrB(-) → {n_glpT4_pmrB1}/{n_total4} = {n_glpT4_pmrB1/n_total4:.2%}")

# ----------------------------
# Simple decision tree on clusters 1 vs 2
# ----------------------------
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Build a combined matrix for clusters 1 and 2 with a 'cluster' label
c = pd.concat([
    amr_clusters[1].assign(cluster=1),
    amr_clusters[2].assign(cluster=2)
], axis=0)

X = c.drop(columns=["cluster"])
y = c["cluster"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
clf = DecisionTreeClassifier(random_state=0, max_depth=5)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Top-20 important features
importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False).head(20)
plt.figure(figsize=(5, 3))
sns.barplot(x=importances.values, y=importances.index)
plt.title("Top 20 Most Important Features (Decision Tree)")
plt.xlabel("Importance"); plt.ylabel("Feature")
plt.tight_layout(); plt.show()

genes_of_interest = ['gyrA_S83L', 'tet(A)', 'sul2', 'blaTEM-1']

def pct_with_any(df: pd.DataFrame, genes: list[str]) -> tuple[float, int, int]:
    """
    Return (percent, numerator, denominator) for genomes that have >=1 of the specified genes.
    Fills missing genes with 0.
    """
    if df.empty:
        return 0.0, 0, 0
    sub = df.reindex(columns=genes, fill_value=0)
    has_any = (sub == 1).any(axis=1)
    n = int(has_any.sum())
    d = len(df)
    pct = (n / d * 100) if d else 0.0
    return pct, n, d

pct_c1, n_c1, d_c1 = pct_with_any(amr_clusters[1], genes_of_interest)
pct_c2, n_c2, d_c2 = pct_with_any(amr_clusters[2], genes_of_interest)

print(
    "Clusters 1 and 2 differed in other AMR gene profiles; "
    f"{pct_c2:.1f}% of Cluster 2 genomes ({n_c2}/{d_c2}) carried at least one of "
    "gyrA S83L, sul2, tet(A), or blaTEM-1, whereas only "
    f"{pct_c1:.1f}% of Cluster 1 genomes ({n_c1}/{d_c1}) harbored any of these genes."
)
