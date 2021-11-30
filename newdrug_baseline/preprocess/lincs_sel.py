#%%
from cmapPy.pandasGEXpress import parse, write_gctx, GCToo
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
lincs_dir = '/rd1/user/tanyh/perturbation/dataset/' 
data_dir = '/rd1/user/tanyh/perturbation/CPA/datasets/'
#pd.set_option('display.max_rows', None)

# cell info
sigFileName = data_dir + 'siginfo_beta.txt'
sigInfo = pd.read_csv(sigFileName, sep = "\t")
sigInfoCP = sigInfo[sigInfo["pert_type"]=='trt_cp']

sigInfoCP = sigInfoCP[sigInfoCP["pert_dose_unit"]=='uM']
sigInfoCP = sigInfoCP.sort_values("pert_time",ascending=False)
sigInfoCP = sigInfoCP.groupby(["pert_dose", "cmap_name", "cell_iname"]).head(1)
sigInfoCP.dropna(axis=0,subset=["pert_dose"],inplace=True)
sigInfoCP.index = sigInfoCP["sig_id"]
CPsig = sigInfoCP["sig_id"]
print(CPsig)

# lincs
lincs_cp = parse.parse(file_path = data_dir+'level5_beta_trt_cp_n720216x12328.gctx', cid = CPsig)
lincs_cp.col_metadata_df["cell"] = sigInfoCP["cell_iname"]
lincs_cp.col_metadata_df["drug"] = sigInfoCP["cmap_name"]
lincs_cp.col_metadata_df["dose"] = sigInfoCP["pert_dose"]
print(lincs_cp.data_df.shape)

train_index, test_index = train_test_split(range(lincs_cp.data_df.shape[1]),test_size=0.01, random_state=0)
lincs_cp_train = GCToo.GCToo(lincs_cp.data_df.iloc[:, train_index], col_metadata_df=lincs_cp.col_metadata_df.iloc[train_index, :])
print(lincs_cp_train.data_df.shape)
print(lincs_cp_train.col_metadata_df.shape)
write_gctx.write(lincs_cp_train, lincs_dir+"trt_sel_train.gctx")
lincs_cp_test =  GCToo.GCToo(lincs_cp.data_df.iloc[:, test_index], col_metadata_df=lincs_cp.col_metadata_df.iloc[test_index, :])
print(lincs_cp_test.data_df.shape)
print(lincs_cp_test.col_metadata_df.shape)
write_gctx.write(lincs_cp_test, lincs_dir+"trt_sel_test.gctx")


# %%
