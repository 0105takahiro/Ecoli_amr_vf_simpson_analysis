#%%
from dataclasses import dataclass
from email.utils import decode_rfc2231
from sqlite3 import DatabaseError
import pandas as pd
import numpy as np
import os
import re
import glob
import subprocess
from subprocess import PIPE
import io
import copy
import time
from threading import Thread
import matplotlib
import random
from collections import Counter
from collections import defaultdict
import pickle

#%%
%cd /Users/sekinetakahiro/reserch/E_coli_Jaccard_and_Simpson_Project/Output/co_occurring_gene_clusters
#%%
with open( 'amr_clusters.pkl', 'rb') as f:
    amr_clusters=pickle.load(f)
with open( 'vf_clusters.pkl', 'rb') as f:
    vf_clusters= pickle.load(f)
#%%
with open( 'represent_gene_amr.pkl', 'rb') as f:
    amr_represent=pickle.load(f)
with open( 'represent_gene_vf.pkl', 'rb') as f:
    vf_represent= pickle.load(f)
# %%
%cd /Users/sekinetakahiro/reserch/E_coli_Jaccard_and_Simpson_Project/Output/gene_gene_distance
amr_jaccard=pd.read_csv('gene_gene_distance_amr_jaccard.csv',index_col=0)
amr_simpson=pd.read_csv('gene_gene_distance_amr_simpson.csv',index_col=0)

# %%
# 前提:
# amr_jaccard … 'Jaccard distance' を持つDF
# amr_simpson … 'Simpson distance' を持つDF
key = ['Gene 1', 'Gene 2', 'Phylogroup']
meta_cols = ['Odds ratio', 'P-value', 'No. of gene 1', 'No. of gene 2']

# 1) キーでインデックスを揃える
jac_idx  = amr_jaccard.set_index(key)
simp_idx = amr_simpson.set_index(key)

# （任意）キー重複の検査
assert not jac_idx.index.duplicated().any(), "amr_jaccard has duplicate keys"
assert not simp_idx.index.duplicated().any(), "amr_simpson has duplicate keys"

# 2) 列方向に結合（共通キーのみ：join='inner'）
out = pd.concat(
    [
        jac_idx[['Jaccard distance'] + meta_cols],   # 距離＋共通メタをJaccard側から採用
        simp_idx[['Simpson distance']],              # Simpson距離だけを追加
    ],
    axis=1, join='inner'
).reset_index()

# 3) 列順の整形（お好みで）
out = out[['Gene 1','Gene 2','Jaccard distance','Simpson distance',
           'Odds ratio','P-value','No. of gene 1','No. of gene 2','Phylogroup']]

# %%
out
# %%
