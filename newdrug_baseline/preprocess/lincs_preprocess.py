#%%
from cmapPy.pandasGEXpress import parse, write_gctx, GCToo
import numpy as np
import pandas as pd
#from k_means_constrained import KMeansConstrained
from sklearn.model_selection import train_test_split
lincs_dir = '/rd1/user/tanyh/perturbation/dataset/' 
data_dir = '/rd1/user/tanyh/perturbation/CPA/datasets/'
#pd.set_option('display.max_rows', None)

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
print(CPsig)

#%% DEBUG
"""print(sigInfoCP['cmap_name'].value_counts())
cell_vocab_path = "/rd1/user/tanyh/perturbation/pretrain_BERT/cell_vocab.txt"
with open(cell_vocab_path) as f:
    cell_vocab = f.read().split('\n')
cell_dict = {cell_name:i for i,cell_name in enumerate(cell_vocab)}
cell_counts = pd.DataFrame(sigInfoCP["cell_iname"].value_counts()).reset_index()
cell_counts.index = cell_counts['index'].apply(lambda x:cell_dict[x])
print(cell_counts)"""

#%% gene info
geneFileName = data_dir + 'geneinfo_beta.txt'
geneInfo = pd.read_csv(geneFileName, sep = "\t")
lmInfo = geneInfo["gene_id"][geneInfo["feature_space"] == "landmark"].astype(str) # landmark genes only
print(geneInfo.shape)
print(lmInfo.shape)
print(lmInfo)
#lmInfo.to_csv("gene_vocab.txt",index=None, header=None)

#%% Save gene_symbol vocab
"""gene_symbol=geneInfo.loc[geneInfo["feature_space"]== "landmark",["gene_symbol"]]
gene_symbol.reset_index(inplace=True,drop=True)
gene_symbol["pos"]=gene_symbol.index#+4
gene_symbol.to_csv("gene_symbol_pos_cv.csv",index=None)"""

#%%
#x = parse.parse(file_path = data_dir+'level5_beta_trt_cp_n720216x12328.gctx', rid = lmInfo)
#write_gctx.write(x, data_dir+'level5_beta_trt_cp_n720216x12328_landmark_only.gctx')
#%%
L1000FileName = data_dir + 'level5_beta_trt_cp_n720216x12328_landmark_only.gctx'
lincs_cp = parse.parse(L1000FileName, cid = CPsig, rid = lmInfo)
lincs_cp.col_metadata_df["cell"] = sigInfoCP["cell_iname"]
lincs_cp.col_metadata_df["drug"] = sigInfoCP["cmap_name"]
lincs_cp.col_metadata_df["dose"] = sigInfoCP["pert_dose"]
print(lincs_cp.data_df.shape)
#write_gctx.write(lincs_cp, "trt_cp_landmarkonly_split.gctx")
#%% sel 82 genes
with open("sciplex_lincs_intersect.txt","r") as f:
    sel_gene = f.read().split('\n')
geneFileName = data_dir + 'geneinfo_beta.txt'
geneInfo = pd.read_csv(geneFileName, sep = "\t")
selInfo = geneInfo["gene_id"][geneInfo["gene_symbol"].isin(sel_gene)].astype(str)
selSym = geneInfo["gene_symbol"][geneInfo["gene_symbol"].isin(sel_gene)].astype(str)
selSym.index = selInfo
print(selInfo)
L1000FileName = data_dir + 'level5_beta_trt_cp_n720216x12328_landmark_only.gctx'
lincs_cp = parse.parse(L1000FileName, cid = CPsig, rid = selInfo)
lincs_cp.col_metadata_df["cell"] = sigInfoCP["cell_iname"]
lincs_cp.col_metadata_df["drug"] = sigInfoCP["cmap_name"]
lincs_cp.col_metadata_df["dose"] = sigInfoCP["pert_dose"]
lincs_cp.row_metadata_df["gene_symbol"] = selSym
print(lincs_cp.data_df.shape)
write_gctx.write(lincs_cp, lincs_dir+"trt_cp_landmarkonly_split_selgene.gctx")

#%%
"""percent=np.zeros((lincs_cp.data_df.shape[0],))
for i in range(lincs_cp.data_df.shape[0]):
    row=lincs_cp.data_df.iloc[i,:]
    percent[i]=row[row>-1][row<1].shape[0]/row.shape[0]
print(percent.min(),percent.max())"""
#%%
lincs_cp = parse.parse(lincs_dir+"trt_cp_landmarkonly_split_selgene.gctx")
train_index, test_index = train_test_split(range(lincs_cp.data_df.shape[1]),test_size=0.01, random_state=0)
lincs_cp_train = GCToo.GCToo(lincs_cp.data_df.iloc[:, train_index], col_metadata_df=lincs_cp.col_metadata_df.iloc[train_index, :])
print(lincs_cp_train.data_df.shape)
print(lincs_cp_train.col_metadata_df.shape)
write_gctx.write(lincs_cp_train, lincs_dir+"trt_cp_landmarkonly_train_selgene.gctx")
lincs_cp_test =  GCToo.GCToo(lincs_cp.data_df.iloc[:, test_index], col_metadata_df=lincs_cp.col_metadata_df.iloc[test_index, :])
print(lincs_cp_test.data_df.shape)
print(lincs_cp_test.col_metadata_df.shape)
write_gctx.write(lincs_cp_test, lincs_dir+"trt_cp_landmarkonly_test_selgene.gctx")

#%% Generate tiny dataset
"""lincs_cp = parse.parse(lincs_dir+"trt_cp_landmarkonly_train.gctx")
drug_counts = lincs_cp.col_metadata_df['drug'].value_counts()
sel_drug=drug_counts[drug_counts>600].index
idx = lincs_cp.col_metadata_df['drug'].isin(sel_drug)
idx = idx.values.nonzero()[0]
lincs_cp_sel = GCToo.GCToo(lincs_cp.data_df.iloc[:, idx], col_metadata_df=lincs_cp.col_metadata_df.iloc[idx, :])

train_index, test_index = train_test_split(range(lincs_cp_sel.data_df.shape[1]),test_size=0.1, random_state=0)
lincs_cp_test =  GCToo.GCToo(lincs_cp_sel.data_df.iloc[:, test_index], col_metadata_df=lincs_cp_sel.col_metadata_df.iloc[test_index, :])
print(lincs_cp_test.data_df.shape)
print(lincs_cp_test.col_metadata_df.shape)
print(lincs_cp_test.col_metadata_df['drug'].value_counts())
write_gctx.write(lincs_cp_test, lincs_dir+"trt_cp_landmarkonly_seldrug_tinytrain.gctx")"""
#%% Generate small dataset
"""lincs_cp = parse.parse(lincs_dir+"trt_cp_landmarkonly_tinytrain.gctx")

train_index, test_index = train_test_split(range(lincs_cp.data_df.shape[1]),test_size=0.005, random_state=0)
lincs_cp_test =  GCToo.GCToo(lincs_cp.data_df.iloc[:, test_index], col_metadata_df=lincs_cp.col_metadata_df.iloc[test_index, :])
print(lincs_cp_test.data_df.shape)
print(lincs_cp_test.col_metadata_df.shape)
write_gctx.write(lincs_cp_test, lincs_dir+"trt_cp_landmarkonly_smalltrain.gctx")"""


#%%
lincs_cp_train = parse.parse(lincs_dir+"trt_cp_landmarkonly_train.gctx")
#lincs_cp_test = parse.parse(lincs_dir+"trt_cp_landmarkonly_test.gctx")
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

#%% Cluster
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
