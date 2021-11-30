#%%
import scanpy as sc
from cmapPy.pandasGEXpress import parse, write_gctx, GCToo
import pandas as pd
from sklearn.model_selection import train_test_split


lincs_dir = '/rd1/user/tanyh/perturbation/dataset/' 
data_dir = '/rd1/user/tanyh/perturbation/CPA/datasets/'

adata = sc.read('/rd1/user/tanyh/perturbation/finetune_ae/datasets/sciplex3_0909_82gene.h5ad')

lincs = parse.parse(lincs_dir+"trt_cp_landmarkonly_split_selgene.gctx")
lincs.row_metadata_df["gene_symbol"].to_csv("selgene_symbol.txt",header=None,index=None)
data_df = lincs.data_df
row_metadata_df = lincs.row_metadata_df
row_metadata_df["id"] = row_metadata_df.index
row_metadata_df.index = row_metadata_df["gene_symbol"]
data_df.index = row_metadata_df.index


lincs_adata = sc.AnnData(data_df.T, lincs.col_metadata_df,row_metadata_df)

print(adata.X.shape,lincs_adata.X.shape)
print(adata.var_names, lincs_adata.var_names)

adata_idx = adata.obs_names
adata_idx_train, adata_idx_test = train_test_split(adata_idx,
                                                test_size=0.01,
                                                random_state=42)
lincs_adata_idx = lincs_adata.obs_names
lincs_adata_idx_train, lincs_adata_idx_test = train_test_split(lincs_adata_idx,
                                                test_size=0.01,
                                                random_state=42)                                                
adata_all = lincs_adata[lincs_adata_idx_test].concatenate(adata[adata_idx_test], batch_categories=['lincs', 'sciplex'])
sc.pp.pca(adata_all)
sc.pp.neighbors(adata_all)
sc.tl.umap(adata_all)
sc.pl.umap(adata_all, color=['batch'], palette=sc.pl.palettes.vega_20_scanpy)
# %%
lincs_cp_train = parse.parse(lincs_dir+"trt_cp_landmarkonly_train_selgene.gctx")
lincs_cp_index = pd.Series(lincs_cp_train.data_df.index)

geneFileName = data_dir + 'geneinfo_beta.txt'
geneInfo = pd.read_csv(geneFileName, sep = "\t")
id2symbol = {id:symbol for id, symbol in zip(geneInfo["gene_id"],geneInfo["gene_symbol"])}
lincs_cp_index = lincs_cp_index.apply(lambda x: id2symbol[int(x)])
print(lincs_cp_index)
# %%
