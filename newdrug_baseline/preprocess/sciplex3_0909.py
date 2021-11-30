#%%
import sys
import os
import pandas as pd

sys.path.append(os.path.join(sys.path[0], '../'))

import numpy as np
from sklearn.model_selection import train_test_split
import scanpy as sc
import helper
import pickle

np.random.seed(0)

#%%
dir = '/rd1/user/tanyh/perturbation/CPA/datasets/sciplex3_old_reproduced.h5ad'

adata = sc.read(dir)
print(adata)
dataset_dir = '/rd1/user/tanyh/perturbation/dataset/aca/'
species = 'human'
test = {'human':['Wang_Kidney'], 'mouse':['Adam']}

with open(dataset_dir+'gene_'+species+'_lst_'+test[species][0]+'.p','rb') as f:
    gene_list=pickle.load(f)

inter = np.intersect1d(adata.var_names,gene_list)
print(inter.size)
#%%
symbol_pos = pd.read_csv("gene_symbol_pos_cv.csv")
print(symbol_pos)

adata.obs.drop([
    'split1', 'split2', 'split3', 'split4', 'split5', 'split6', 'split7',
    'split8', 'split9', 'split10', 'split11', 'split12', 'split13', 'split14',
    'split15', 'split16', 'split17', 'split18', 'split19', 'split20',
    'split21', 'split22', 'split23', 'split24', 'split25', 'split26',
    'split27', 'split28', 'old_split', 'split_all', 'split'
],axis=1, inplace=True)

with open("selgene_symbol.txt",'r') as f:
    sel_gene = f.read().split('\n')
print(sel_gene)

#with open("sciplex_lincs_intersect.txt","w") as f:
#    f.write('\n'.join(sel_gene))

adata = adata[:,sel_gene].copy()
print(adata)

helper.rank_genes_groups_by_cov(adata,
                                groupby='cov_drug',
                                covariate='cell_type',
                                control_group='control',
                                n_genes=50,
                                key_added='rank_genes_groups_cov',
                                largestd=True)
new_genes_dict = {}
for cat in adata.obs.cov_drug_dose_name.unique():
    if 'control' not in cat:
        rank_keys = np.array(list(adata.uns['rank_genes_groups_cov'].keys()))
        bool_idx = [x in cat for x in rank_keys]
        genes = adata.uns['rank_genes_groups_cov'][rank_keys[bool_idx][0]]
        new_genes_dict[cat] = genes
adata.uns['rank_genes_groups_cov'] = new_genes_dict

helper.rank_genes_groups_by_cov(adata,
                                groupby='cov_drug',
                                covariate='cell_type',
                                control_group='control',
                                n_genes=10,
                                key_added='rank_genes_groups_cov_10',
                                largestd=True)
new_genes_dict = {}
for cat in adata.obs.cov_drug_dose_name.unique():
    if 'control' not in cat:
        rank_keys = np.array(list(
            adata.uns['rank_genes_groups_cov_10'].keys()))
        bool_idx = [x in cat for x in rank_keys]
        genes = adata.uns['rank_genes_groups_cov_10'][rank_keys[bool_idx][0]]
        new_genes_dict[cat] = genes
adata.uns['rank_genes_groups_cov_10'] = new_genes_dict

#print('control cov_drug_dose_name',set(adata.obs.loc[adata.obs['control'] == 1, 'cov_drug_dose_name']))
#print('Number of control', adata.obs.groupby('control').size())

sc.pp.scale(adata)
adata.var.drop(['num_cells_expressed-0-0', 'num_cells_expressed-1-0', 
    'num_cells_expressed-1', 'highly_variable', 'means', 'dispersions', 
    'dispersions_norm', 'mean', 'std'],axis=1,inplace=True)

"""zero_var = pd.DataFrame(None, index=np.setdiff1d(symbol_pos["gene_symbol"].values,sel_gene))
all_zero = sc.AnnData(np.zeros((adata.shape[0],978-len(sel_gene))),adata.obs,zero_var,adata.uns)

adata = sc.concat([adata, all_zero],axis=1,join='outer',merge='unique',uns_merge='unique')
adata = adata[:,symbol_pos["gene_symbol"].values]"""


split_name = 'drug'
adata.obs[split_name] = 'nan'
drugs_name = np.unique(adata.obs['condition'])
seen_drugs_num = int(drugs_name.shape[0] * 0.7)
seen_drug = np.random.choice(drugs_name, seen_drugs_num, replace=False)
print('Choose', len(seen_drug), 'seen cov_drug.', seen_drug)

adata.obs.loc[adata.obs['control'] == 1, split_name] = 'traintest'
adata.obs.loc[adata.obs['condition'].isin(seen_drug),
              split_name] = 'traintest'
adata_idx = adata.obs_names[adata.obs[split_name] == 'traintest']
adata_idx_train, adata_idx_test = train_test_split(adata_idx,
                                                test_size=0.1,
                                                random_state=42)
adata.obs.loc[adata_idx_train, split_name] = 'train'
adata.obs.loc[adata_idx_test, split_name] = 'test'
adata.obs.loc[adata.obs[split_name] == 'nan', split_name] = 'ood'
print(adata.obs[split_name].value_counts())

adata.write('/rd1/user/tanyh/perturbation/finetune_ae/datasets/sciplex3_0909_82gene.h5ad')

# %%
adata = sc.read('/rd1/user/tanyh/perturbation/finetune_ae/datasets/sciplex3_0817.h5ad')
print(adata)
# %%
"""import matplotlib.pyplot as plt
plt.hist(adata.X.flatten())"""

# %%
from cmapPy.pandasGEXpress import parse

lincs_dir = '/rd1/user/tanyh/perturbation/dataset/' 
lincs_cp_train = parse.parse(lincs_dir+"trt_cp_landmarkonly_train.gctx")
lincs_drug = lincs_cp_train.col_metadata_df["drug"]

adata_ood = adata[adata.obs["drug"]=="ood"]
adata_ood_lincs = adata_ood.obs["name_in_lincs"]

# %%
adata_unique = np.unique(adata_ood_lincs)
lincs_unique = np.unique(lincs_drug)
adata_not_in_lincs = np.setdiff1d(adata_unique,lincs_unique)
print(adata_not_in_lincs)
# %%
