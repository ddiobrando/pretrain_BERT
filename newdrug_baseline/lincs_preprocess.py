#%%
from cmapPy.pandasGEXpress import parse, write_gctx
import numpy as np
import pandas as pd

data_dir = '/rd1/user/tanyh/perturbation/CPA/datasets/'

#%% cell info
sigFileName = data_dir + 'siginfo_beta.txt'
sigInfo = pd.read_csv(sigFileName, sep = "\t")
sigInfoCP = sigInfo[sigInfo["pert_type"]=='trt_cp']

sigInfoCP = sigInfoCP[sigInfoCP["pert_dose_unit"]=='uM']
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

# %%
