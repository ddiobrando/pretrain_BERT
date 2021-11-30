import torch
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.join(sys.path[0], '../../datasets'))
from disentangled_data_loader import DisentangledDataLoader

DRUG_EXPRESSION_DATA_LOC=''


MIN_NUM_TIMES_PER_UNSEEN_SEQ = 6
NUM_SEEN_TIMES_PER_UNSEEN_SEQ = 2  # don't include 0 in this


def generate_unseen_drug_cell(num_unseen_combos, train_df):
    # don't want to drop any dose=0
    drug_ids = train_df[train_df['dosage'] != 0]
    drug_ids = list(drug_ids['drug_id'].drop_duplicates(inplace=False))
    candidates = {}
    for drug in drug_ids:
        # figure out how many unique cells it maps to
        cells = train_df[train_df['drug_id'] == drug]['cell_id'].drop_duplicates(inplace=False)
        #if len(cells) > 1:
        #    for cell in cells:
        #        # make sure I have the minimum number of times!
        #        dosages = train_df[train_df['drug_id'] == drug]
        #        dosages = dosages[dosages['cell_id'] == cell]
        #        dosages = dosages['dosage'].drop_duplicates()
        #        if len(dosages) >= MIN_NUM_TIMES_PER_UNSEEN_SEQ:
        #            if drug not in candidates:
        #                candidates[drug] = []
        #            candidates[drug].append(cell)
        for cell in cells:
            #print(cell)
            # make sure I have the minimum number of times!
            dosages = train_df[train_df['drug_id'] == drug]
            dosages = dosages[dosages['cell_id'] == cell]
            dosages = dosages['dosage'].drop_duplicates()
            if drug not in candidates:
                candidates[drug] = []
            candidates[drug].append(cell)

    # randomly generate the combos.
    unseen_combos = set()
    unseen_drugs = np.random.choice([drug for drug in candidates], size=num_unseen_combos, replace=False)
    for uDrug in unseen_drugs:
        uCell = np.random.choice(candidates[uDrug], size=1)[0]
        unseen_combos.add((uDrug, uCell))
    return unseen_combos


def generate_unseen_dosages(unseen_combos, train_df):
    unseen_dosages = set()
    for uDrug, uCell in unseen_combos:
        relevant_df = train_df[train_df['drug_id'] == uDrug]
        relevant_df = relevant_df[relevant_df['cell_id'] == uCell]
        # choose T - 3. Don't choose 0.
        relevant_df = relevant_df[relevant_df['dosage'] != 0]
        unique_dosages = np.unique(relevant_df['dosage'])
        uDose = np.random.choice(unique_dosages, size=len(unique_dosages) - NUM_SEEN_TIMES_PER_UNSEEN_SEQ)
        for d in uDose:
            unseen_dosages.add(d)
    return unseen_dosages


def load_data():
    df = pd.read_csv(DRUG_EXPRESSION_DATA_LOC + 'pert_doseattribute_less2.txt', delimiter='\t', header=None)
    df.columns = ["cell_id", "drug_id", "dosage"]
    df_exprs = pd.read_csv(DRUG_EXPRESSION_DATA_LOC + 'pert_doseexpression_less2.txt', delimiter='\t', header=None)
    df['expression'] = df_exprs.values.tolist()

    return df


class DrugExpressionDosageDataset(torch.utils.data.Dataset):
    def __init__(self, train: bool = True, train_df=None, device=torch.device('cuda:0')):
        super().__init__()
        print('...loading data')
        self.df = load_data()

        self.drugs_to_use = np.unique(self.df['drug_id'])
        self.cells_to_use = np.unique(self.df['cell_id'])

        self.dosages_to_use = sorted(list(np.unique(self.df['dosage'])))
        self.dosages_idx = {self.dosages_to_use[i]: i for i in range(len(self.dosages_to_use))}

        print('\t...constructing mappings')
        self.drug_id_to_idx_mapping = {drug: i for (i, drug) in enumerate(
            self.drugs_to_use)}
        self.cell_id_to_idx_mapping = {cell: i for (i, cell) in enumerate(
            self.cells_to_use)}

        if train:
            self.df = self.df[:int(len(self.df) * 0.8)]
        else:
            # take the rows that _aren't_ part of train
            train_idxs = train_df.index
            self.df = self.df.drop(index=train_idxs)

        print('\t...determining combos')
        # now, need to group based on (drug id, cell id) combinations -> concentration handled as continuous index!
        self.drug_cell_combos = self.df[['drug_id', 'cell_id']]
        self.drug_cell_combos = self.drug_cell_combos.drop_duplicates()

        self.N = len(self.drug_cell_combos)
        self.device = device
        print('...done with constructor')

    def get_drugs(self):
        return self.drugs_to_use

    def get_cells(self):
        return self.cells_to_use

    def num_drugs(self):
        return len(self.drugs_to_use)

    def num_cells(self):
        return len(self.cells_to_use)

    def get_drug_encoding(self, drug_id):
        enc = torch.zeros(len(self.drug_id_to_idx_mapping))
        enc[self.drug_id_to_idx_mapping[drug_id]] = 1
        return enc

    def get_cell_encoding(self, cell_id):
        enc = torch.zeros(len(self.cell_id_to_idx_mapping))
        enc[self.cell_id_to_idx_mapping[cell_id]] = 1
        return enc

    def __getitem__(self, item):
        rowOI = self.drug_cell_combos.iloc[item]
        drug_id = rowOI['drug_id']
        cell_id = rowOI['cell_id']

        df_OI = self.df[self.df['drug_id'] == drug_id]
        df_OI = df_OI[df_OI['cell_id'] == cell_id]
        df_OI = df_OI.sort_values(by='dosage', ascending=True)  # use increasing dosages

        drug_one_hot = self.get_drug_encoding(drug_id)
        cell_one_hot = self.get_cell_encoding(cell_id)
        dosages_arr = df_OI['dosage']

        gene_expr = torch.stack(
            [torch.tensor(df_OI.iloc[i]['expression']) for i in range(len(df_OI))], dim=0)

        #if len(dosages_arr) != len(np.unique(dosages_arr)):
        #    print('dosages are:', df_OI['dosage'])
        #    raise ValueError(
        #        'Odd result for drug {} and cell {}'.format(drug_id, cell_id))

        mask = torch.zeros(len(self.dosages_to_use))
        mask[[self.dosages_idx[dos] for dos in dosages_arr]] = 1

        full_expr = torch.zeros((len(self.dosages_to_use), gene_expr.shape[-1]), dtype=torch.float)
        full_expr[[self.dosages_idx[dos] for dos in dosages_arr]] = gene_expr

        # now, dosages is just arange
        dosages = torch.tensor(self.dosages_to_use, dtype=float)

        return drug_one_hot, cell_one_hot, dosages, full_expr, mask

    def __len__(self):
        return self.N


class DrugExpressionDosageDatasetWithUnseen(DrugExpressionDosageDataset):
    def __init__(self, train: bool = True, unseen: set = None, num_unseen_combos: int = None,
                 use_unseen_dosages: bool = False, unseen_dosages: set = None, 
                 train_df=None, device=torch.device('cuda:0')):
        super().__init__(train, train_df=train_df, device=device)
        if num_unseen_combos is not None:
            unseen = generate_unseen_drug_cell(num_unseen_combos, self.df)
        if unseen is None:
            unseen = set()
        if use_unseen_dosages:
            if train:
                unseen_dosages = generate_unseen_dosages(unseen, self.df)
            else:
                assert unseen_dosages is not None
        else:
            unseen_dosages = set()
        self.unseen = unseen
        self.unseen_dosages = unseen_dosages

        for cell_id, drug_id in unseen:
            self.df = self.df[(self.df['cell_id'] != cell_id) | (self.df['drug_id'] != drug_id)]
        for dosage in self.unseen_dosages:
            self.df = self.df[self.df['dosage'] != dosage]

        self.drug_cell_combos = self.df[['drug_id', 'cell_id']]
        self.drug_cell_combos = self.drug_cell_combos.drop_duplicates()

        self.N = len(self.drug_cell_combos)

    def get_unseen(self):
        return self.unseen

    def get_unseen_dosages(self):
        return self.unseen_dosages


class DrugExpressionDosageDatasetWithUnseenForCVAE(DrugExpressionDosageDatasetWithUnseen):
    def __init__(self, train: bool = True, unseen: set = None, num_unseen_combos: int = None,
                 use_unseen_dosages: bool = False, unseen_dosages: set = None, 
                 train_df=None, device=torch.device('cuda:0')):
        super().__init__(train, unseen, num_unseen_combos, use_unseen_dosages, unseen_dosages, 
                         train_df=train_df, device=device)

    def __getitem__(self, item):
        drug_one_hot, cell_one_hot, dosages, expression, mask = super().__getitem__(item)
        T = len(dosages)
        repeated_drug = torch.stack([drug_one_hot for _ in range(T)])
        repeated_cell = torch.stack([cell_one_hot for _ in range(T)])

        return repeated_drug, repeated_cell, dosages, expression, mask


class DrugExpressionDosageDataLoader(DisentangledDataLoader):
    def __init__(self, device=torch.device('cuda:0'), val_split: float = 0.2, create_datasets=True):
        if create_datasets:
            self.train_dataset = DrugExpressionDosageDataset(
                train=True, device=device)
            self.drugs_to_use = self.train_dataset.get_drugs()
            self.cells_to_use = self.train_dataset.get_cells()
            
            self.test_dataset = DrugExpressionDosageDataset(
                train=False, train_df=self.train_dataset.df, device=device)
            
            len_data = len(self.train_dataset)
            train_n = int((1 - val_split) * len_data)
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                self.train_dataset, lengths=[train_n, len_data - train_n])
        self.batch_size = 1024  # by default

    def get_drug_list(self):
        return self.drugs_to_use

    def get_cell_list(self):
        return self.cells_to_use

    def plot_image(self, image, labels, filename=None):
        raise NotImplementedError()

    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size

    def get_batch_size(self):
        return self.batch_size

    def get_train_dataset(self):
        return self.train_dataset

    def get_validation_dataset(self):
        return self.val_dataset

    def get_test_dataset(self):
        return self.test_dataset

    def get_data_loader(self, dataset_split):
        if dataset_split == 'train':
            dataset = self.get_train_dataset()
        elif dataset_split == 'val':
            dataset = self.get_validation_dataset()
        elif dataset_split == 'test':
            dataset = self.get_test_dataset()
        else:
            raise ValueError('Dataset split must be train, val, or test!')
        return DataLoader(dataset, batch_size=self.get_batch_size(), shuffle=False, num_workers=0)


class DrugExpressionDosageWithUnseenDataLoader(DrugExpressionDosageDataLoader):
    def __init__(self, device=torch.device('cuda:0'), unseen: set = None, num_unseen_combos: int = None,
                 use_unseen_dosages: bool = False, unseen_dosages: set = None, val_split: float = 0.2,
                 create_datasets=True):
        super().__init__(device=device, val_split=val_split, create_datasets=False)
        if create_datasets:
            self.train_dataset = DrugExpressionDosageDatasetWithUnseen(
                train=True, device=device, unseen=unseen, num_unseen_combos=num_unseen_combos,
                use_unseen_dosages=use_unseen_dosages, unseen_dosages=unseen_dosages)
            self.drugs_to_use = self.train_dataset.get_drugs()
            self.cells_to_use = self.train_dataset.get_cells()
            self.unseen_combos = self.train_dataset.get_unseen()
            self.unseen_dosages = self.train_dataset.get_unseen_dosages()
            
            self.test_dataset = DrugExpressionDosageDatasetWithUnseen(
                train=False, device=device, unseen=self.unseen_combos, 
                unseen_dosages=self.unseen_dosages, train_df=self.train_dataset.df)

            len_data = len(self.train_dataset)
            train_n = int((1 - val_split) * len_data)
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                self.train_dataset, lengths=[train_n, len_data - train_n])

    def get_unseen_combos(self):
        return self.unseen_combos

    def get_unseen_dosages(self):
        return self.unseen_dosages


class DrugExpressionDosageWithUnseenDataLoaderForCVAE(DrugExpressionDosageWithUnseenDataLoader):
    def __init__(self, device=torch.device('cuda:0'), unseen: set = None, num_unseen_combos: int = None,
                 use_unseen_dosages: bool = False, unseen_dosages: set = None, val_split: float = 0.2,
                 create_datasets=True):
        super().__init__(device=device, unseen=unseen, num_unseen_combos=num_unseen_combos,
                         use_unseen_dosages=use_unseen_dosages, unseen_dosages=unseen_dosages,
                         val_split=val_split, create_datasets=False)
        if create_datasets:
            self.train_dataset = DrugExpressionDosageDatasetWithUnseenForCVAE(
                train=True, device=device, unseen=unseen, num_unseen_combos=num_unseen_combos,
                use_unseen_dosages=use_unseen_dosages, unseen_dosages=unseen_dosages)
            self.drugs_to_use = self.train_dataset.get_drugs()
            self.cells_to_use = self.train_dataset.get_cells()
            self.unseen_combos = self.train_dataset.get_unseen()
            self.unseen_dosages = self.train_dataset.get_unseen_dosages()
            
            self.test_dataset = DrugExpressionDosageDatasetWithUnseenForCVAE(
                train=False, device=device, unseen=self.unseen_combos, 
                unseen_dosages=self.unseen_dosages, train_df=self.train_dataset.df)

            len_data = len(self.train_dataset)
            train_n = int((1 - val_split) * len_data)
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                self.train_dataset, lengths=[train_n, len_data - train_n])
