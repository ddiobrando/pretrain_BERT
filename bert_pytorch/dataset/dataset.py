import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
import random
from cmapPy.pandasGEXpress import parse
import pdb


class BERTDataset(Dataset):
    def __init__(self, data_path, cell_vocab,drug_vocab):
        """Remove encoding="utf-8", corpus_lines=None, on_memory=True
        """
        self.cell_vocab = cell_vocab
        self.drug_vocab = drug_vocab
        data = parse.parse(data_path)
        self.data_df = data.data_df.values.T
        self.seq_len = self.data_df.shape[1]
        self.cell = data.col_metadata_df["cell"].values
        self.drug = data.col_metadata_df["drug"].values
        self.dose = data.col_metadata_df["dose"].values
        cell_dict = {cell_name:i for i,cell_name in enumerate(self.cell_vocab)}
        self.cell_serie = pd.Series(cell_dict)
        drug_dict = {drug_name:i for i,drug_name in enumerate(self.drug_vocab)}
        self.drug_serie = pd.Series(drug_dict)


    def __len__(self):
        return self.data_df.shape[0]

    def __getitem__(self, item):
        t1, cell, drug, dose = self.data_df[item], self.cell[item], self.drug[item], self.dose[item]
        t1, t1_label = self.random_number(t1)
        bert_input, bert_label = t1.unsqueeze(0), t1_label.unsqueeze(0)
        #t1_list, t1_label_list = [], []
        #for value in self.kmeans_label.values():
        #    pad = torch.nn.ConstantPad1d(padding=(0, self.seq_len - len(value)), value=-666)
        #    t1_list.append(pad(t1[value][:self.seq_len]))
        #    t1_label_list.append(pad(t1_label[value][:self.seq_len]))

        #bert_input = torch.stack(t1_list,dim=0)
        #bert_label = torch.stack(t1_label_list, dim=0)

        cell_label = self.cell_serie[cell]
        drug_label = self.drug_serie[drug]
        dose_label = torch.tensor([dose],dtype=torch.float32)

        #randkey = random.sample(self.kmeans_label.keys(),1)[0]
        #t1, t1_label = self.random_number(t1[self.kmeans_label[randkey]])

        #bert_input = t1[:self.seq_len]
        #bert_label = t1_label[:self.seq_len]

        #cell_label = torch.zeros(len(self.cell_vocab),dtype=torch.long)
        #cell_label[self.cell_serie[cell]] = 1
        #drug_label = torch.zeros(len(self.drug_vocab),dtype=torch.long)
        #drug_label[self.drug_serie[drug]] = 1

        #pad = torch.nn.ConstantPad1d(padding=(0, self.seq_len - len(bert_input)), value=-666)
        #bert_input = pad(bert_input)
        #bert_label = pad(bert_label)
        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  #"segment_label": segment_label,
                  "cell": cell_label,
                  "drug": drug_label,
                  "dose": dose_label}

        return output #{key: torch.tensor(value) for key, value in output.items()}

    def random_number(self, sentence):
        """ 
        bert_input: token
        bert_label: sentence
        """
        tokens = sentence[:]

        for i, token in enumerate(sentence):
            prob = random.random()
            if prob < 0.15:
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

            #else:
            #    tokens[i] = token

        return torch.Tensor(tokens), torch.Tensor(sentence)

