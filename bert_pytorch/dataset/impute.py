import random
import torch
from torch.utils.data import Dataset
from cmapPy.pandasGEXpress import parse

import numpy as np
import pandas as pd

class ImputeDataset(Dataset):
    def __init__(self, data_path, cell_vocab,drug_vocab, gene_thre): #, gene_vocab):
        """Remove encoding="utf-8", corpus_lines=None, on_memory=True
        """
        self.cell_vocab = cell_vocab
        self.drug_vocab = drug_vocab
        self.special = ["PAD","CLS","MASK","UNK"]
        self.pad_index = 0
        self.cls_index = 1
        self.mask_index = 2
        self.unk_index = 3
        data = parse.parse(data_path)
        self.data_df = data.data_df.values.T
        self.seq_len = 512
        self.gen_len = self.data_df.shape[1]+len(self.special)
        self.cell = data.col_metadata_df["cell"].values
        self.drug = data.col_metadata_df["drug"].values
        self.dose = np.log(data.col_metadata_df["dose"].values).astype(np.float32)
        cell_dict = {cell_name:i for i,cell_name in enumerate(self.cell_vocab)}
        self.cell_serie = pd.Series(cell_dict)
        drug_dict = {drug_name:i for i,drug_name in enumerate(self.drug_vocab)}
        self.drug_serie = pd.Series(drug_dict)
        self.gene_thre = gene_thre
        assert self.gene_thre.shape[0]==99, "Expect gene_thre to be 1%, ... 99%" 

    def __len__(self):
        return self.data_df.shape[0]

    def __getitem__(self, item):
        t1 = self.data_df[item]
        bert_input, bert_label = self.random_number(t1)

        output = {"bert_input": bert_input,
                  "bert_label": bert_label}

        return {key: torch.tensor(value) for key, value in output.items()}

    def random_number(self, tokens):
        """ 
        bert_input: token
        bert_label: sentence
        """
        output_label = []

        for i in range(len(tokens)):
            prob = random.random()
            if prob < 0.15:
                output_label.append(tokens[i])
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = 666

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.uniform(-10,10)

                # 10% randomly change token to current token
                #else:
                #    tokens[i] = token

            else:
                output_label.append(0)
                #tokens[i] = token

        return torch.Tensor(tokens), torch.Tensor(output_label)