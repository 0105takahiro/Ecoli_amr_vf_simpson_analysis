#%%
from pathlib import Path
import random
import pickle
import pandas as pd
import networkx as nx
from docplex.mp.model import Model
#%%
%cd /Users/sekinetakahiro/reserch/E_coli_Jaccard_and_Simpson_Project/Output/co_occurring_gene_clusters
with open("amr_clusters_2.pkl", "rb") as f:
    amr_clusters = pickle.load(f)
with open("vf_clusters_2.pkl", "rb") as f:
    vf_clusters = pickle.load(f)

# %%
amr_clusters
# %%
