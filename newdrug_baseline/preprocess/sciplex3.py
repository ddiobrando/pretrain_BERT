#%%
import sys
import os

sys.path.append(os.path.join(sys.path[0], '../'))

import numpy as np
from sklearn.model_selection import train_test_split
import scanpy as sc
import helper

np.random.seed(0)
dir = '/rd1/user/tanyh/perturbation/CPA/datasets/sciplex3_old_reproduced.h5ad'

adata = sc.read(dir)
#%%
adata.obs.drop([
    'split1', 'split2', 'split3', 'split4', 'split5', 'split6', 'split7',
    'split8', 'split9', 'split10', 'split11', 'split12', 'split13', 'split14',
    'split15', 'split16', 'split17', 'split18', 'split19', 'split20',
    'split21', 'split22', 'split23', 'split24', 'split25', 'split26',
    'split27', 'split28', 'old_split', 'split_all', 'split'
],axis=1, inplace=True)


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

adata.write('/rd1/user/tanyh/perturbation/CPA/datasets/sciplex3_0817.h5ad')

# %%
