#%%
from cmapPy.pandasGEXpress import parse, write_gctx, GCToo
import numpy as np
import pandas as pd
#from k_means_constrained import KMeansConstrained
from sklearn.model_selection import train_test_split
data_dir = '/rd1/user/tanyh/perturbation/CPA/datasets/'

#%% cell info
sigFileName = data_dir + 'siginfo_beta.txt'
sigInfo = pd.read_csv(sigFileName, sep = "\t")
sigInfoCP = sigInfo[sigInfo["pert_type"]=='trt_cp']

sigInfoCP = sigInfoCP[sigInfoCP["pert_dose_unit"]=='uM']
sigInfoCP = sigInfoCP.sort_values("pert_time",ascending=False)
sigInfoCP = sigInfoCP.groupby(["pert_dose", "cmap_name", "cell_iname"]).head(1)
sigInfoCP.dropna(axis=0,subset=["pert_dose"],inplace=True)
sigInfoCP.index = sigInfoCP["sig_id"]
CPsig = sigInfoCP["sig_id"]

# gene info
geneFileName = data_dir + 'geneinfo_beta.txt'
geneInfo = pd.read_csv(geneFileName, sep = "\t")
lmInfo = geneInfo["gene_id"][geneInfo["feature_space"] == "landmark"].astype(str) # landmark genes only
print(geneInfo.shape)
print(lmInfo.shape)
lmInfo.to_csv("gene_vocab.txt",index=None, header=None)


#x = parse.parse(file_path = 'level5_beta_trt_cp_n720216x12328.gctx', rid = lmInfo)
#write_gctx.write(x, 'level5_beta_trt_cp_n720216x12328_landmark_only.gctx')
L1000FileName = data_dir + 'level5_beta_trt_cp_n720216x12328_landmark_only.gctx'
lincs_cp = parse.parse(L1000FileName, cid = CPsig, rid = lmInfo)
lincs_cp.col_metadata_df["cell"] = sigInfoCP["cell_iname"]
lincs_cp.col_metadata_df["drug"] = sigInfoCP["cmap_name"]
lincs_cp.col_metadata_df["dose"] = sigInfoCP["pert_dose"]
print(lincs_cp.data_df.shape)
#write_gctx.write(lincs_cp, "trt_cp_landmarkonly_split.gctx")
#%%
"""percent=np.zeros((lincs_cp.data_df.shape[0],))
for i in range(lincs_cp.data_df.shape[0]):
    row=lincs_cp.data_df.iloc[i,:]
    percent[i]=row[row>-1][row<1].shape[0]/row.shape[0]
print(percent.min(),percent.max())"""
#%%
lincs_cp = parse.parse("trt_cp_landmarkonly_split.gctx")
train_index, test_index = train_test_split(range(lincs_cp.data_df.shape[1]),test_size=0.01, random_state=0)
lincs_cp_train = GCToo.GCToo(lincs_cp.data_df.iloc[:, train_index], col_metadata_df=lincs_cp.col_metadata_df.iloc[train_index, :])
print(lincs_cp_train.data_df.shape)
print(lincs_cp_train.col_metadata_df.shape)
#write_gctx.write(lincs_cp_train, "trt_cp_landmarkonly_train.gctx")
lincs_cp_test =  GCToo.GCToo(lincs_cp.data_df.iloc[:, test_index], col_metadata_df=lincs_cp.col_metadata_df.iloc[test_index, :])
print(lincs_cp_test.data_df.shape)
print(lincs_cp_test.col_metadata_df.shape)
#write_gctx.write(lincs_cp_test, "trt_cp_landmarkonly_test.gctx")

#%%
#lincs_cp_train = parse.parse("trt_cp_landmarkonly_train.gctx")
lincs_cp_test = parse.parse("trt_cp_landmarkonly_test.gctx")
#print(np.max(lincs_cp_train.col_metadata_df["dose"]))
#print(np.max(lincs_cp_test.col_metadata_df["dose"]))

#%%
for i, row in sigInfoCP.iterrows():
    if np.isnan(row["pert_dose"]):
        print(row)
#%%
X=lincs_cp.data_df
print(X.shape)
print(X.dropna().shape)

#%%
"""
num_clusters = 40
X=lincs_cp.data_df
#kmeans = KMeansConstrained(n_clusters=num_clusters,size_min=5,size_max=40,random_state=0).fit(X)
#np.save("kmeans_label.npy",kmeans.labels_)
#%%
kmeans_labels = np.load("kmeans_label.npy")
labels = [[] for _ in range(num_clusters)]
for i in range(len(kmeans_labels)):
    labels[kmeans_labels[i]].append(i)
#%%
data_df=pd.concat([X.iloc[labels[i], :].reset_index(drop=True) for i in range(num_clusters)], axis=1)
col_metadata_df = pd.concat([lincs_cp.col_metadata_df for _ in range(num_clusters)])
data_df.columns = list(range(data_df.shape[1]))
col_metadata_df.reset_index(drop=True, inplace=True)
lincs_rec = GCToo.GCToo(data_df, col_metadata_df=col_metadata_df)
write_gctx.write(lincs_rec, "trt_cp_landmarkonly_split.gctx")
"""
# %%
