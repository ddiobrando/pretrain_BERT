# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import warnings
import torch
import pdb
import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)
import scanpy as sc
import pandas as pd

from sklearn.preprocessing import OneHotEncoder


def ranks_to_df(data, key='rank_genes_groups'):
    """Converts an `sc.tl.rank_genes_groups` result into a MultiIndex dataframe.

    You can access various levels of the MultiIndex with `df.loc[[category]]`.

    Params
    ------
    data : `AnnData`
    key : str (default: 'rank_genes_groups')
        Field in `.uns` of data where `sc.tl.rank_genes_groups` result is
        stored.
    """
    d = data.uns[key]
    dfs = []
    for k in d.keys():
        if k == 'params':
            continue
        series = pd.DataFrame.from_records(d[k]).unstack()
        series.name = k
        dfs.append(series)

    return pd.concat(dfs, axis=1)

def get_drug_encoding(self, drug_id):
    enc = torch.zeros(len(self.drug_id_to_idx_mapping))
    enc[self.drug_id_to_idx_mapping[drug_id]] = 1
    return enc

def get_cell_encoding(self, cell_id):
    enc = torch.zeros(len(self.cell_id_to_idx_mapping))
    enc[self.cell_id_to_idx_mapping[cell_id]] = 1
    return enc


class Dataset:
    def __init__(self,
                 fname,
                 perturbation_key,
                 dose_key,
                 cell_type_key,
                 split_key='split',
                 model='cpa',
                 cell_vocab_path='../lincs_ae/cell_vocab.txt',
                 drug_vocab_path='../lincs_ae/drug_vocab.txt', 
                 logdose = True):
        data = sc.read(fname)

        self.model = model
        self.perturbation_key = perturbation_key
        self.dose_key = dose_key
        self.cell_type_key = cell_type_key
        if self.model == 'codegen':
            # Change the dose of control to 0
            self.control_name = data.obs.loc[data.obs['control']==1, perturbation_key][0]
            print('control_name', self.control_name)
            data.obs.loc[data.obs[perturbation_key] == self.control_name,
                         dose_key] = 0

        self.var_names = data.var_names
        self.pert_categories = np.array(data.obs['cov_drug_dose_name'].values)

        self.de_genes = data.uns['rank_genes_groups_cov']
        self.de_genes_10 = data.uns['rank_genes_groups_cov_10'] if 'rank_genes_groups_cov_10' in data.uns else None

        #print("*** Please change single_drug in data.py if you have multiple drugs. ***")
        self.single_drug = False

        if model in ['AE']:
            sc.pp.scale(data)
            print("Normalize data ...")

            self.perturbation_key="name_in_lincs"
            print("*** Changing perturbation_key to name_in_lincs. ***")
            print("Loading Cell Vocab", cell_vocab_path)
            with open(cell_vocab_path) as f:
                cell_vocab = f.read().split('\n')
            print("Vocab Size: ", len(cell_vocab))

            print("Loading Drug Vocab", drug_vocab_path)
            with open(drug_vocab_path) as f:
                drug_vocab = f.read().split('\n')
            print("Vocab Size: ", len(drug_vocab))

            cell_dict = {cell_name:i for i,cell_name in enumerate(cell_vocab)}
            self.cell_serie = pd.Series(cell_dict)
            drug_dict = {drug_name:i for i,drug_name in enumerate(drug_vocab)}
            self.drug_serie = pd.Series(drug_dict)

            try:
                self.genes = torch.Tensor(data.X.A)
            except:
                self.genes = torch.Tensor(data.X)

            self.drugs_names = np.array(data.obs[perturbation_key].values)
            self.dose_names = np.log1p(data.obs[dose_key].values).astype(np.float32)
            self.cell_types_names = np.array(data.obs[cell_type_key].values)

            self.num_cell_types = len(cell_vocab)
            self.num_genes = self.genes.shape[1]
            self.num_drugs = len(drug_vocab)

        elif model != 'codegen':
            try:
                self.genes = torch.Tensor(data.X.A)
            except:
                self.genes = torch.Tensor(data.X)
            if model == 'BERT':
                print("*** Preprocess self.genes with the same methods in aca. ***")
                self.genes = self.genes / (self.genes.sum(axis=1, keepdims=True) / 10000)
                self.genes = np.log1p(self.genes)
            self.drugs_names = np.array(data.obs[perturbation_key].values)
            self.dose_names = np.array(data.obs[dose_key].values)
            self.cell_types_names = np.array(data.obs[cell_type_key].values)

            if self.single_drug:
                self.drugs_names_unique = np.unique(self.drugs_names)
            else:
                # get unique drugs
                drugs_names_unique = set()
                for d in self.drugs_names:
                    [drugs_names_unique.add(i) for i in d.split("+")]
                self.drugs_names_unique = np.array(list(drugs_names_unique))
            self.cell_types_names_unique = np.unique(self.cell_types_names)

            self.num_cell_types = len(self.cell_types_names_unique)
            self.num_genes = self.genes.shape[1]
            self.num_drugs = len(self.drugs_names_unique)
            #self.ctrl = data.obs['control'].values
            # save encoder for a comparison with Mo's model
            # later we need to remove this part
            encoder_drug = OneHotEncoder(sparse=False)
            encoder_drug.fit(self.drugs_names_unique.reshape(-1, 1))

            self.atomic_drugs_dict = dict(zip(self.drugs_names_unique, encoder_drug.transform(
                        self.drugs_names_unique.reshape(-1, 1))))

            encoder_ct = OneHotEncoder(sparse=False)
            encoder_ct.fit(self.cell_types_names_unique.reshape(-1, 1))

            if self.single_drug:
                drug = torch.Tensor(
                    encoder_drug.transform(self.drugs_names.reshape(
                        -1, 1))).float()
                if (model != 'cpa') and logdose:
                    dose = np.log(data.obs[dose_key].values).astype(np.float32)
                else:
                    dose = data.obs[dose_key].values
                self.drugs = drug * np.repeat(np.expand_dims(dose,1), drug.shape[1],axis=1)
            else:
                # get drug combinations
                drugs = []
                for i, comb in enumerate(self.drugs_names):
                    drugs_combos = encoder_drug.transform(
                        np.array(comb.split("+")).reshape(-1, 1))
                    dose_combos = str(data.obs[dose_key].values[i]).split("+")
                    if (model != 'cpa') and logdose:
                        for j, d in enumerate(dose_combos):
                            if j == 0:
                                drug_ohe = np.log(float(d)) * drugs_combos[j]
                            else:
                                drug_ohe += np.log(float(d)) * drugs_combos[j]
                    else:
                        for j, d in enumerate(dose_combos):
                            if j == 0:
                                drug_ohe = float(d) * drugs_combos[j]
                            else:
                                drug_ohe += float(d) * drugs_combos[j]
                    drugs.append(drug_ohe)
                self.drugs = torch.Tensor(drugs)

            self.atomic_сovars_dict = dict(
                zip(
                    list(self.cell_types_names_unique),
                    encoder_ct.transform(
                        self.cell_types_names_unique.reshape(-1, 1))))

            self.cell_types = torch.Tensor(
                encoder_ct.transform(self.cell_types_names.reshape(
                    -1, 1))).float()

        else:
            self.df = pd.DataFrame({
                'drug_id': data.obs[perturbation_key].values,
                'cell_id': data.obs[cell_type_key].values,
                'dosage': data.obs[dose_key].values,
                'expression': data.X.A.tolist()
            })
            self.drug_cell_combos = data.obs[[perturbation_key, cell_type_key]]
            self.drug_cell_combos = self.drug_cell_combos.drop_duplicates()
            drugs_names_unique = np.unique(self.df['drug_id'])
            cell_types_names_unique = np.unique(self.df['cell_id'])
            self.drug_id_to_idx_mapping = {
                            drug: i
                            for (i, drug) in enumerate(drugs_names_unique)
                        }
            self.cell_id_to_idx_mapping = {
                    cell: i
                    for (i, cell) in enumerate(cell_types_names_unique)
                }
            self.dosages_to_use = sorted(np.unique(data.obs[dose_key].values))
            self.dosages_idx = {
                self.dosages_to_use[i]: i
                for i in range(len(self.dosages_to_use))
            }
            self.num_cell_types = len(cell_types_names_unique)
            self.num_genes = data.X.A.shape[1]
            self.num_drugs = len(drugs_names_unique)
            self.mean_size = int(self.df.groupby(['drug_id', 'cell_id', 'dosage']).count().mean()['expression'])
            self.N = len(self.drug_cell_combos) * (self.mean_size//20 + 1)

        self.indices = {
            "all": list(range(data.X.shape[0])),
            "control": np.where(data.obs['control'] == 1)[0].tolist(),
            "treated": np.where(data.obs['control'] != 1)[0].tolist(),
            "train": np.where(data.obs[split_key] == 'train')[0].tolist(),
            "test": np.where(data.obs[split_key] == 'test')[0].tolist(),
            "ood": np.where(data.obs[split_key] == 'ood')[0].tolist(),
            #"continuous": np.where(data.obs[split_key] == 'continuous')[0].tolist(),
            #"combination": np.where(data.obs[split_key] == 'combination')[0].tolist(),
            #"contcomb": np.where(data.obs[split_key] == 'contcomb')[0].tolist()
        }

    def subset(self, split, condition="all"):
        idx = list(set(self.indices[split]) & set(self.indices[condition]))
        return SubDataset(self, idx, self.model)

    def __getitem__(self, i):
        if self.model in ['AE']:
            gene, drug_name, cell_name, dose =  self.genes[i], self.drugs_names[i], self.cell_types_names[i], self.dose_names[i]
            drug, cell = self.onehot(drug_name, cell_name, dose)
            return gene, drug, cell
            
        elif self.model != 'codegen':
            return self.genes[i], self.drugs[i], self.cell_types[i]

        else:
            rowOI = self.drug_cell_combos.iloc[i % len(self.drug_cell_combos)]
            drug_id = rowOI[self.perturbation_key]
            cell_id = rowOI[self.cell_type_key]

            df_OI = self.df[self.df['cell_id'] == cell_id]
            df_OI = df_OI[(df_OI['drug_id'] == drug_id) |
                          (df_OI['drug_id'] == self.control_name)]
            sel_index = []
            for dosage, df_dosage in df_OI.groupby('dosage'):
                sel_index.append(np.random.choice(df_dosage.index, 1)[0])
            df_OI = df_OI.loc[sel_index, :]
            df_OI = df_OI.sort_values(by='dosage',
                                      ascending=True)  # use increasing dosages
            drug_one_hot = get_drug_encoding(self, drug_id)
            cell_one_hot = get_cell_encoding(self, cell_id)
            dosages_arr = df_OI['dosage']

            gene_expr = torch.stack([
                torch.tensor(df_OI.iloc[i]['expression'])
                for i in range(len(df_OI))
            ],
                                    dim=0)

            mask = torch.zeros(len(self.dosages_to_use))
            mask[[self.dosages_idx[dos] for dos in dosages_arr]] = 1

            full_expr = torch.zeros(
                (len(self.dosages_to_use), gene_expr.shape[-1]),
                dtype=torch.float)
            full_expr[[self.dosages_idx[dos]
                       for dos in dosages_arr]] = gene_expr

            # now, dosages is just arange
            dosages = torch.tensor(self.dosages_to_use, dtype=float)

            T = dosages.shape[0]
            repeated_drug = torch.stack([drug_one_hot for _ in range(T)])
            repeated_cell = torch.stack([cell_one_hot for _ in range(T)])
            return full_expr, repeated_drug, dosages, repeated_cell, mask

    def __len__(self):
        if self.model != 'codegen':
            return len(self.genes)
        else:
            return self.N

    def onehot(self, drug_name, cell_name, dose):
        if self.single_drug:
            drug = torch.zeros(self.num_drugs)
            if drug_name in self.drug_serie.index:
                drug[self.drug_serie[drug_name]]=float(dose)
            else:
                drug[self.drug_serie['unknown']]=float(dose)
        else:
            # get drug combinations
            drug = torch.zeros(self.num_drugs)
            for drug, dose in zip(drug_name.split("+"),
                                    str(dose).split("+")):
                if drug in self.drug_serie.index:
                    drug[self.drug_serie[drug_name]]=float(dose)
                else:
                    drug[self.drug_serie['unknown']]=float(dose)

        cell = torch.zeros(self.num_cell_types,dtype=torch.int)
        if cell_name in self.cell_serie.index:
            cell[self.cell_serie[cell_name]] = 1
        else:
            cell[self.cell_serie["unknown"]] =1
        return drug, cell


class SubDataset:
    """
    Subsets a `Dataset` by selecting the examples given by `indices`.
    """
    def __init__(self, dataset, indices, model='cpa'):
        self.model = model

        self.var_names = dataset.var_names
        self.de_genes = dataset.de_genes
        self.de_genes_10 = dataset.de_genes_10
        #self.ctrl_name = dataset.ctrl_name[0]
        self.pert_categories = dataset.pert_categories[indices]
        self.num_cell_types = dataset.num_cell_types
        self.num_genes = dataset.num_genes
        self.num_drugs = dataset.num_drugs
        self.single_drug = dataset.single_drug

        if model in ['AE']:
            self.genes = dataset.genes[indices]
            self.drugs_names = dataset.drugs_names[indices]
            self.cell_types_names = dataset.cell_types_names[indices]
            self.dose_names = dataset.dose_names[indices]
            self.cell_serie = dataset.cell_serie
            self.drug_serie = dataset.drug_serie

            self.perturbation_key = dataset.perturbation_key
            self.dose_key = dataset.dose_key
            self.cell_type_key = dataset.cell_type_key
        elif model != 'codegen':
            self.genes = dataset.genes[indices]
            self.drugs = dataset.drugs[indices]
            self.cell_types = dataset.cell_types[indices]
            #self.drugs_names = dataset.drugs_names[indices]
            #self.cell_types_names = dataset.cell_types_names[indices]

            self.perturbation_key = dataset.perturbation_key
            self.dose_key = dataset.dose_key
            self.cell_type_key = dataset.cell_type_key

            #self.perts_dict = dataset.atomic_drugs_dict
            #self.covars_dict = dataset.atomic_сovars_dict

        else:
            self.perturbation_key = 'drug_id'
            self.dose_key = 'dosage'
            self.cell_type_key = 'cell_id'

            self.drug_id_to_idx_mapping = dataset.drug_id_to_idx_mapping
            self.cell_id_to_idx_mapping = dataset.cell_id_to_idx_mapping

            self.drug_cell_combos = dataset.drug_cell_combos
            self.df = dataset.df.loc[indices, :]
            self.drug_id_to_idx_mapping = dataset.drug_id_to_idx_mapping
            self.cell_id_to_idx_mapping = dataset.cell_id_to_idx_mapping
            self.drug_cell_combos = self.df[['drug_id','cell_id']]
            self.drug_cell_combos = self.drug_cell_combos.drop_duplicates()
            self.dosages_to_use = sorted(np.unique(self.df['dosage']))
            self.dosages_idx = {
                self.dosages_to_use[i]: i
                for i in range(len(self.dosages_to_use))
            }
            self.control_name = dataset.control_name
            self.mean_size = int(self.df.groupby(['drug_id', 'cell_id', 'dosage']).count().mean()['expression'])
            self.N = len(self.drug_cell_combos) * (self.mean_size//20 + 1)

    def __getitem__(self, i):
        return Dataset.__getitem__(self, i)

    def __len__(self):
        return Dataset.__len__(self)

    def onehot(self, drug_name, cell_name, dose):
        return Dataset.onehot(self, drug_name, cell_name, dose)


def load_dataset_splits(dataset_path,
                        perturbation_key,
                        dose_key,
                        cell_type_key,
                        split_key,
                        return_dataset=False,
                        model='cpa', 
                        logdose=True):

    dataset = Dataset(dataset_path, perturbation_key, dose_key, cell_type_key,
                      split_key, model, logdose)

    splits = {
        "training": dataset.subset("train", "all"),
        "training_control": dataset.subset("train", "control"),
        "training_treated": dataset.subset("train", "treated"),
        "test": dataset.subset("test", "all"),
        "test_control": dataset.subset("test", "control"),
        "test_treated": dataset.subset("test", "treated"),
        "ood": dataset.subset("ood", "all"),
        #"continuous": dataset.subset("continuous", "all"),
        #"combination": dataset.subset("combination", "all"),
        #"contcomb": dataset.subset("contcomb", "all")
    }

    if return_dataset:
        return splits, dataset
    else:
        return splits
