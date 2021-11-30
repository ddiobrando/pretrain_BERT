#%%
import numpy as np
from sklearn.model_selection import train_test_split
import scanpy as sc

np.random.seed(0)
adata = sc.read('../datasets/GSM_new_split.h5ad')
print('Number of control',adata.obs.groupby('control').size())

#i=1
#for ood_dose_num in [3,4,5]:
#i=4
#for ood_dose_num in [2]:
i=7
for ood_dose_num in [5]:
    split_name = f'split{i}'
    adata.obs[split_name] = 'nan'
    adata.obs['ct_drug'] = adata.obs['cell_type'].astype(str)+'_'+adata.obs['drug'].astype(str)
    ood_condition = []

    for ct_drug, group in adata.obs[adata.obs['control']==0].groupby('ct_drug'):
        ood_dose = np.random.choice(group['dose_val'].drop_duplicates(), ood_dose_num, replace=False)
        ood_condition.extend([ct_drug+'_'+str(dose) for dose in ood_dose])

    print('Choose', len(ood_condition),'ood condition:',ood_condition)

    adata.obs.loc[
        adata.obs['cov_drug_dose_name'].isin(ood_condition),split_name
    ] = 'ood'

    adata_idx = adata.obs_names[adata.obs[split_name]!='ood']
    adata_idx_train, adata_idx_test = train_test_split(adata_idx, test_size=0.4, random_state=42)
    adata.obs.loc[adata_idx_train, split_name] = 'train'
    adata.obs.loc[adata_idx_test, split_name] = 'test'
    print(adata.obs.groupby(split_name).size())
    i+=1

adata.write('../datasets/GSM_new_split.h5ad')
# %%
