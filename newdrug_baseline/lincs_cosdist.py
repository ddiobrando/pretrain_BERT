#%%
from cmapPy.pandasGEXpress import parse, write_gctx
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from data import Dataset

data_dir = '/rd1/user/tanyh/perturbation/CPA/datasets/'
#%%
adata = Dataset(data_dir + 'sciplex3_1016.h5ad')

#%% cell info
sigFileName = data_dir + 'siginfo_beta.txt'
sigInfo = pd.read_csv(sigFileName, sep = "\t")
sigInfoCP = sigInfo[sigInfo["pert_type"]=='trt_cp']

cmap_lincs_sciplex = set(adata.drugs_names) & set(sigInfoCP["cmap_name"])
sigInfoCP = sigInfoCP[sigInfoCP["cmap_name"].isin(cmap_lincs_sciplex)] # (44138,35)
sigInfoCP = sigInfoCP[sigInfoCP["pert_dose_unit"]=='uM'] # (44129,35)
CPsig = sigInfoCP["sig_id"]

#%% gene info
geneFileName = data_dir + 'geneinfo_beta.txt'
geneInfo = pd.read_csv(geneFileName, sep = "\t")
lmInfo = geneInfo["gene_id"][geneInfo["feature_space"] == "landmark"].astype(str) # landmark genes only
print(geneInfo.shape)
print(lmInfo.shape)

#%%
#x = parse.parse(file_path = 'level5_beta_trt_cp_n720216x12328.gctx', rid = gid)
#write_gctx.write(x, 'level5_beta_trt_cp_n720216x12328_landmark_only.gctx')
L1000FileName = data_dir + 'level5_beta_trt_cp_n720216x12328_landmark_only.gctx'
lincs_cp = parse.parse(L1000FileName, cid = CPsig, rid = lmInfo)
print(lincs_cp.data_df.shape)

#%%
for cell_type in np.unique(adata.cell_types_names):
    sigCell = sigInfoCP[sigInfoCP["cell_iname"]==cell_type]
    sigCell = sigCell.sort_values("pert_time",ascending=False)
    sigCell = sigCell.groupby(["pert_dose", "cmap_name"]).head(1)

    for dosage in np.unique(adata.dose_names):
        # dosage is 0.1* of real dosage (uM)
        sigCellDose = sigCell[(sigCell["pert_dose"]>=5*dosage)&
            (sigCell["pert_dose"]<=50*dosage)]
        print(cell_type, dosage, sigCellDose.shape)

        cmap_names = []
        cmap_exp = []
        for cmap, cmap_meta in sigCellDose.groupby('cmap_name'):
            cmap_lincs = lincs_cp.data_df.loc[:,cmap_meta['sig_id'].values]
            print(cmap, cmap_lincs.shape)
            cmap_names.append(cmap)
            cmap_exp.append(cmap_lincs.mean(axis=1).values)
        cmap_num = len(cmap_names)
        dist = np.zeros((cmap_num, cmap_num))
        for i in range(cmap_num):
            for j in range(cmap_num):
                dist[i][j]=cosine(cmap_exp[i], cmap_exp[j])
        dist_df = pd.DataFrame(dist,index=cmap_names,columns=cmap_names)
        dist_df.to_csv(cell_type+'_'+str(dosage)+'_cosine.csv')
# %%
