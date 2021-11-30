#%%
import sys
import os
import pandas as pd

sys.path.append(os.path.join(sys.path[0], '../'))

import numpy as np
from sklearn.model_selection import train_test_split
import scanpy as sc
import helper

np.random.seed(0)

from cmapPy.pandasGEXpress import parse

dir = '/rd1/user/tanyh/perturbation/CPA/datasets/sciplex3_old_reproduced.h5ad'

#%%
adata = sc.read(dir)

drug_vocab_path='../../finetune_old/preprocess/lincs_sciplex3_drug.txt'
with open(drug_vocab_path) as f:
    drug_vocab = f.read().split('\n') + ["unk"]
drug_vocab.remove("DMSO")
print(adata)
#symbol_pos = pd.read_csv("gene_symbol_pos_cv.csv")
#print(symbol_pos)

adata.obs.drop([
    'split1', 'split2', 'split3', 'split4', 'split5', 'split6', 'split7',
    'split8', 'split9', 'split10', 'split11', 'split12', 'split13', 'split14',
    'split15', 'split16', 'split17', 'split18', 'split19', 'split20',
    'split21', 'split22', 'split23', 'split24', 'split25', 'split26',
    'split27', 'split28', 'old_split', 'split_all',
],axis=1, inplace=True)

#with open("selgene_symbol.txt",'r') as f:
#    sel_gene = f.read().split('\n')
#print(sel_gene)

#with open("sciplex_lincs_intersect.txt","w") as f:
#    f.write('\n'.join(sel_gene))

#adata = adata[:,sel_gene].copy()
#print(adata)

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

#sc.pp.scale(adata)

adata.var.drop(['num_cells_expressed-0-0', 'num_cells_expressed-1-0', 
    'num_cells_expressed-1', 'highly_variable', 'means', 'dispersions', 
    'dispersions_norm'],axis=1,inplace=True)

split_name = 'drug'
adata.obs[split_name] = 'nan'
drugs_name = np.unique(adata.obs['name_in_lincs'])
unseen_drugs_num = int(drugs_name.shape[0] * 0.3)
unseen_drug = np.random.choice(np.intersect1d(drugs_name,drug_vocab), unseen_drugs_num, replace=False)
print('Choose', len(unseen_drug), 'unseen drug.', unseen_drug)

adata.obs.loc[adata.obs['control'] == 1, split_name] = 'traintest'
adata.obs.loc[adata.obs['name_in_lincs'].isin(unseen_drug),
              split_name] = 'ood'
adata.obs.loc[adata.obs[split_name] == 'nan', split_name] = 'traintest'
adata_idx = adata.obs_names[adata.obs[split_name] == 'traintest']
adata_idx_train, adata_idx_test = train_test_split(adata_idx,
                                                test_size=0.1,
                                                random_state=42)
adata.obs.loc[adata_idx_train, split_name] = 'train'
adata.obs.loc[adata_idx_test, split_name] = 'test'

print(adata.obs[split_name].value_counts())

adata.write('/rd1/user/tanyh/perturbation/CPA_modified/datasets/sciplex3_1016.h5ad')

# %%
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
adata = sc.read('/rd1/user/tanyh/perturbation/CPA_modified/datasets/sciplex3_1016.h5ad')
print(pd.crosstab(adata.obs.name_in_lincs, adata.obs.drug))

drug_vocab_path='../../finetune_old/preprocess/lincs_sciplex3_drug.txt'
with open(drug_vocab_path) as f:
    drug_vocab = f.read().split('\n') + ["unk"]
adata_drug = np.unique(adata.obs.loc[adata.obs["drug"]=="ood","name_in_lincs"])
assert "DMSO" not in adata_drug
print(len(adata_drug))
print(len(np.intersect1d(drug_vocab, adata_drug)))
# %%
