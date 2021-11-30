from collections import defaultdict
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
import random
from cmapPy.pandasGEXpress import parse
import pdb
import sys


class BERTDataset(Dataset):
    def __init__(self, data_path):
        """Remove encoding="utf-8", corpus_lines=None, on_memory=True
        """
        #self.cell_vocab = cell_vocab
        #self.drug_vocab = drug_vocab
        #self.seq_len = seq_len
        data = parse.parse(data_path)
        self.data_df = data.data_df.values.T
        self.seq_len = self.data_df.shape[1]
        #self.cell = data.col_metadata_df["cell"].values
        #self.drug = data.col_metadata_df["drug"].values
        #self.dose = np.log(data.col_metadata_df["dose"].values).astype(np.float32)
        #cell_dict = {cell_name:i for i,cell_name in enumerate(self.cell_vocab)}
        #self.cell_serie = pd.Series(cell_dict)
        #drug_dict = {drug_name:i for i,drug_name in enumerate(self.drug_vocab)}
        #self.drug_serie = pd.Series(drug_dict)


    def __len__(self):
        return self.data_df.shape[0]//512*512

    def __getitem__(self, item):
        item = np.random.choice(self.data_df.shape[0])
        t1 = self.data_df[item]
        #t1, cell, drug, dose = self.data_df[item], self.cell[item], self.drug[item], self.dose[item]
        bert_input, bert_label = self.random_number(t1)

        #cell_label = self.cell_serie[cell]
        #drug_label = self.drug_serie[drug]
        #dose_label = torch.tensor([dose])

        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  #"cell": cell_label,
                  #"drug": drug_label,
                  #"dose": dose_label
                  }

        return output

    def random_number(self, tokens):
        output_label = tokens.copy()

        mask_pos = np.random.choice(len(tokens),int(0.8*len(tokens)),replace=False)
        tokens[mask_pos]=0

        return torch.Tensor(tokens), torch.Tensor(output_label)
