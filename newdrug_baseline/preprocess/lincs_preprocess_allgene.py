#%%
from cmapPy.pandasGEXpress import parse, write_gctx, GCToo
import numpy as np
import pandas as pd
import scanpy as sc
#from k_means_constrained import KMeansConstrained
from sklearn.model_selection import train_test_split
lincs_dir = '/rd1/user/tanyh/perturbation/dataset/' 
data_dir = '/rd1/user/tanyh/perturbation/CPA/datasets/'
#pd.set_option('display.max_rows', None)

#%%
L1000FileName = data_dir + 'level5_beta_trt_cp_n720216x12328.gctx'
lincs_cp = parse.parse(L1000FileName)

train_index, test_index = train_test_split(range(lincs_cp.data_df.shape[1]),test_size=0.01, random_state=0)
lincs_cp_train = GCToo.GCToo(lincs_cp.data_df.iloc[:, train_index])
print(lincs_cp_train.data_df.shape)
write_gctx.write(lincs_cp_train, lincs_dir+"trt_cp_train.gctx")
lincs_cp_test =  GCToo.GCToo(lincs_cp.data_df.iloc[:, test_index])
print(lincs_cp_test.data_df.shape)
write_gctx.write(lincs_cp_test, lincs_dir+"trt_cp_test.gctx")


# %%
L1000FileName = lincs_dir + 'trt_cp_train.gctx'
lincs_cp = parse.parse(L1000FileName)

# %%
