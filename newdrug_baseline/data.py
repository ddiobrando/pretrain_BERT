# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import warnings
import torch
import pdb
import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)
import scanpy as sc
import pandas as pd

class Dataset:
    def __init__(self,
                 fname,
                 perturbation_key="name_in_lincs",
                 dose_key="dose_val",
                 cell_type_key="cell_type",
                 split_key='drug'):
        """Attribute: pert_categories, de_genes, de_genes_10, genes, 
        drugs_names, dose_names, cell_types_names, indices (tyh update on 8/6)
        """
        data = sc.read(fname)
        #self.perturbation_key = perturbation_key
        #self.dose_key = dose_key
        #self.cell_type_key = cell_type_key

        # Change the dose of control to 0
        # self.control_name = data.obs.loc[data.obs['control']==1, perturbation_key][0]
        # data.obs.loc[data.obs[perturbation_key] == self.control_name, dose_key] = 0

        self.pert_categories = np.array(data.obs['cov_drug_dose_name'].values)

        self.de_genes = data.uns['rank_genes_groups_cov']
        self.de_genes_10 = data.uns['rank_genes_groups_cov_10'] if 'rank_genes_groups_cov_10' in data.uns else None
        self.var_names = data.var_names
        self.genes = torch.Tensor(data.X.A)
        self.drugs_names = np.array(data.obs[perturbation_key].values)
        self.dose_names = np.array(data.obs[dose_key].values)
        self.cell_types_names = np.array(data.obs[cell_type_key].values)

        self.cmap2condition = {}
        condition_names = np.array(data.obs['condition'])
        for condition, cmap in zip(condition_names, self.drugs_names):
            self.cmap2condition[cmap] = condition
        # get unique drugs
        #drugs_names_unique = set()
        #for d in self.drugs_names:
        #    [drugs_names_unique.add(i) for i in d.split("+")]
        #self.drugs_names_unique = np.array(list(drugs_names_unique))
        #self.cell_types_names_unique = np.unique(self.cell_types_names)

        #self.num_cell_types = len(self.cell_types_names_unique)
        #self.num_genes = self.genes.shape[1]
        #self.num_drugs = len(self.drugs_names_unique)


        self.indices = {
            "all": list(range(data.X.shape[0])),
            #"control": np.where(data.obs['control'] == 1)[0].tolist(),
            #"treated": np.where(data.obs['control'] != 1)[0].tolist(),
            "train": np.where(data.obs[split_key] == 'train')[0].tolist(),
            "test": np.where(data.obs[split_key] == 'test')[0].tolist(),
            "ood": np.where(data.obs[split_key] == 'ood')[0].tolist()
        }


    def subset(self, split, condition="all"):
        if split == "train":
            idx = list((set(self.indices["train"])| set(self.indices["test"]))& set(self.indices[condition]))
        elif split == "test":
            idx = list(set(self.indices["ood"])& set(self.indices[condition]))
        return SubDataset(self, idx)

    def __getitem__(self, i):
        return self.genes[i], self.drugs[i], self.cell_types[i]

    def __len__(self):
        return len(self.genes)



class SubDataset:
    """
    Subsets a `Dataset` by selecting the examples given by `indices`.
    Attribute: pert_categories, de_genes, 
        de_genes_10, genes, drugs_names, dose_names, cell_types_names
    """
    def __init__(self, dataset, indices):
        self.pert_categories = dataset.pert_categories[indices]
        self.de_genes = dataset.de_genes
        self.de_genes_10 = dataset.de_genes_10
        self.var_names = dataset.var_names
        self.genes = dataset.genes[indices]
        self.drugs_names = dataset.drugs_names[indices]
        self.dose_names = dataset.dose_names[indices]
        self.cell_types_names = dataset.cell_types_names[indices]
        self.cmap2condition = dataset.cmap2condition

    def __getitem__(self, i):
        return self.genes[i], self.drugs[i], self.cell_types[i]

    def __len__(self):
        return len(self.genes)


def load_dataset_splits(dataset_path,
                        perturbation_key="name_in_lincs",
                        dose_key="dose_val",
                        cell_type_key="cell_type",
                        split_key='drug',
                        return_dataset=False):

    dataset = Dataset(dataset_path, perturbation_key, dose_key, cell_type_key,
                      split_key)

    splits = {
        "train": dataset.subset("train", "all"),
        "test": dataset.subset("test", "all")
    }

    if return_dataset:
        return splits, dataset
    else:
        return splits
