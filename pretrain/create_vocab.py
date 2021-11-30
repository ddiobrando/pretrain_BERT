#%%
from cmapPy.pandasGEXpress import parse, write_gctx, GCToo
import numpy as np
import pandas as pd
data_dir = '/rd1/user/tanyh/perturbation/CPA/datasets/'

# cell info
sigFileName = data_dir + 'siginfo_beta.txt'
sigInfo = pd.read_csv(sigFileName, sep = "\t")
sigInfoCP = sigInfo[sigInfo["pert_type"]=='trt_cp']

sigInfoCP = sigInfoCP[sigInfoCP["pert_dose_unit"]=='uM']
sigInfoCP = sigInfoCP.sort_values("pert_time",ascending=False)
sigInfoCP = sigInfoCP.groupby(["pert_dose", "cmap_name", "cell_iname"]).head(1)
sigInfoCP.index = sigInfoCP["sig_id"]
CPsig = sigInfoCP["sig_id"]
sigInfoCP["cmap_name"].drop_duplicates().to_csv("drug_vocab.txt", header=False, index=None)
sigInfoCP["cell_iname"].drop_duplicates().to_csv("cell_vocab.txt", header=False, index=None)


# %%
