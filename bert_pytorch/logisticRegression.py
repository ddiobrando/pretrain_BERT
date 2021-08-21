import argparse
import os
import pdb
import random
import torch
import time
import tqdm
import json

#from sklearn.linear_model import LogisticRegression, LinearRegression
import numpy as np

import pandas as pd
from torch.utils.data import Dataset
from cmapPy.pandasGEXpress import parse
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn

def pjson(s):
    """
    Prints a string in JSON format and flushes stdout
    """
    print(json.dumps(s), flush=True)


class BERTDataset(Dataset):
    def __init__(self, data_path, cell_vocab,drug_vocab):
        """Remove encoding="utf-8", corpus_lines=None, on_memory=True
        """
        self.cell_vocab = cell_vocab
        self.drug_vocab = drug_vocab
        data = parse.parse(data_path)
        self.data_df = data.data_df.values.T
        self.seq_len = self.data_df.shape[1]
        self.cell = data.col_metadata_df["cell"]
        self.drug = data.col_metadata_df["drug"]
        self.dose = np.log(data.col_metadata_df["dose"].values)
        cell_dict = {cell_name:i for i,cell_name in enumerate(self.cell_vocab)}
        self.cell_serie = pd.Series(cell_dict)
        drug_dict = {drug_name:i for i,drug_name in enumerate(self.drug_vocab)}
        self.drug_serie = pd.Series(drug_dict)
        self.cell_label = self.cell.apply(lambda x: self.cell_serie[x]).values
        self.drug_label = self.drug.apply(lambda x: self.drug_serie[x]).values
        self.data_df = torch.Tensor(self.data_df)
        self.cell_label = torch.Tensor(self.cell_label).long()
        self.drug_label = torch.Tensor(self.drug_label).long()
        self.dose = torch.Tensor(self.dose)

    def __len__(self):
        return self.data_df.shape[0]

    def __getitem__(self, item):
        return self.data_df[item], self.cell_label[item], self.drug_label[item],self.dose[item]


#LayerModel
class MLP(torch.nn.Module):
    """
    A multilayer perceptron with ReLU activations and optional BatchNorm.
    """

    def __init__(self, sizes, batch_norm=True, last_layer_act="linear"):
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers += [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 2 else None,
                torch.nn.ReLU()
            ]

        layers = [l for l in layers if l is not None][:-1]
        self.activation = last_layer_act
        if self.activation == "linear":
            pass
        elif self.activation == "ReLU":
            self.relu = torch.nn.ReLU()
        else:
            raise ValueError("last_layer_act must be one of 'linear' or 'ReLU'")

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        if self.activation == "ReLU":
           x = self.network(x)
           dim = x.size(1) // 2
           return torch.cat((self.relu(x[:, :dim]), x[:, dim:]), dim=1)
        return self.network(x)


def train():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--train_dataset", default="/rd1/user/tanyh/perturbation/pretrain_BERT/trt_cp_landmarkonly_train.gctx", type=str, help="train dataset for train bert")
    parser.add_argument("-t", "--test_dataset", type=str, default="/rd1/user/tanyh/perturbation/pretrain_BERT/trt_cp_landmarkonly_test.gctx", help="test set for evaluate train set")
    parser.add_argument("-cv", "--cell_vocab_path", default="/rd1/user/tanyh/perturbation/pretrain_BERT/cell_vocab.txt", type=str, help="built vocab model path with bert-vocab")
    parser.add_argument("-dv", "--drug_vocab_path", default="/rd1/user/tanyh/perturbation/pretrain_BERT/drug_vocab.txt", type=str, help="built vocab model path with bert-vocab")
    parser.add_argument("-o", "--output_path", default="/rd1/user/tanyh/perturbation/pretrain_BERT/output/0817/", type=str, help="ex)output/")

    parser.add_argument("-b", "--batch_size", type=int, default=512, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=5, help="dataloader worker size")


    args = parser.parse_args()
    if not args.output_path.endswith('/'):
        args.output_path += '/'
    args.log_name = args.output_path
    print(args)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    seed=0
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)


    print("Loading Cell Vocab", args.cell_vocab_path)
    with open(args.cell_vocab_path) as f:
        cell_vocab = f.read().split('\n')
    print("Vocab Cell Size: ", len(cell_vocab))
    
    print("Loading Drug Vocab", args.drug_vocab_path)
    with open(args.drug_vocab_path) as f:
        drug_vocab = f.read().split('\n')
    print("Vocab Drug Size: ", len(drug_vocab))

    print("Loading Train Dataset", args.train_dataset)
    train_dataset = BERTDataset(args.train_dataset, cell_vocab,drug_vocab)
    args.seq_len = train_dataset.seq_len

    print("Loading Test Dataset", args.test_dataset)
    test_dataset = BERTDataset(args.test_dataset, cell_vocab,drug_vocab) \
        if args.test_dataset is not None else None

    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers) \
        if test_dataset is not None else None
    
    

    cellModel = MLP([args.seq_len, len(cell_vocab)]).to('cuda')
    drugModel = MLP([args.seq_len, len(drug_vocab)]).to('cuda')
    doseModel = MLP([args.seq_len, 1]).to('cuda')

    cellOptim = Adam(cellModel.parameters(), lr=1e-3, betas=(0.9,0.999), weight_decay=0.01)
    drugOptim = Adam(drugModel.parameters(), lr=1e-3, betas=(0.9,0.999), weight_decay=0.01)
    doseOptim = Adam(doseModel.parameters(), lr=1e-3, betas=(0.9,0.999), weight_decay=0.01)

    nll = nn.NLLLoss(ignore_index=0)
    mse = nn.MSELoss(reduction='mean')
    for e in range(20):
        data_iter = tqdm.tqdm(enumerate(train_data_loader),
                            total=len(train_data_loader),
                            bar_format="{l_bar}{r_bar}")
        for i, data in data_iter:
            gene, cell, drug, dose = data
            gene=gene.to('cuda')
            cell=cell.to('cuda')
            drug=drug.to('cuda')
            dose=dose.to('cuda')
            cell_pred = cellModel.forward(gene)
            drug_pred = drugModel.forward(gene)
            dose_pred = doseModel.forward(gene)
            
            cellOptim.zero_grad()
            cellLoss = nll(cell_pred,cell)
            cellLoss.backward()
            cellOptim.step()
            
            drugOptim.zero_grad()
            drugLoss = nll(drug_pred,drug)
            drugLoss.backward()
            drugOptim.step()

            doseOptim.zero_grad()
            doseLoss = mse(dose_pred, dose)
            doseLoss.backward()
            doseOptim.step()
        if e % 5==0:
            torch.save(cellModel.cpu(), f"../output/0818regression/cell{e}.pth")
            cellModel.to('cuda')
            torch.save(drugModel.cpu(), f"../output/0818regression/drug{e}.pth")
            drugModel.to('cuda')
            torch.save(doseModel.cpu(), f"../output/0818regression/dose{e}.pth")
            doseModel.to('cuda')

            data_iter = tqdm.tqdm(enumerate(test_data_loader),
                                    total=len(test_data_loader),
                                    bar_format="{l_bar}{r_bar}")
            cellModel.eval()
            drugModel.eval()
            doseModel.eval()
            total_correct = np.zeros(2)
            total_element = np.zeros(2)
            total_dose_loss = 0.0
            total_dose_element = 0
            for i, data in data_iter:
                gene, cell, drug, dose = data
                gene=gene.to('cuda')
                cell=cell.to('cuda')
                drug=drug.to('cuda')
                dose=dose.to('cuda')
                cell_output = cellModel.forward(gene)
                drug_output = drugModel.forward(gene)
                dose_output = doseModel.forward(gene)
                dose_loss = mse(dose_output, dose)

                for k,(output, class_type) in enumerate(zip([cell_output, drug_output], [cell, drug])):
                    correct = output.argmax(dim=-1).eq(class_type).sum().item()
                    total_correct[k] += correct
                    total_element[k] += class_type.nelement()
                total_dose_loss += dose_loss.item()*dose.nelement()
                total_dose_element += dose.nelement()
            print('-----------------------')
            pjson({"epoch":e,
                "total_acc":(total_correct * 100.0 / total_element).tolist(),
                "total_dose_rmse":(total_dose_loss/total_dose_element)**0.5})

    # ------ Can't use sklearn implement because of out of memory ------
    #celllr = LogisticRegression(random_state=0).fit(train_dataset.data_df, train_dataset.cell_label)
    #cell_score = celllr.score(test_dataset.data_df, test_dataset.cell_label)
    #print('cell type mean accrucy:',cell_score)

    #druglr = LogisticRegression(random_state=0).fit(train_dataset.data_df, train_dataset.drug_label)
    #drug_score = druglr.score(test_dataset.data_df, test_dataset.drug_label)
    #print('drug mean accrucy:',drug_score)

    #doselr = LinearRegression().fit(train_dataset.data_df, train_dataset.dose)
    #dose_rmse = np.sqrt(np.mean((doselr.predict(test_dataset.data_df) - test_dataset.dose)**2))
    #print('dose rmse', dose_rmse)


if __name__ == "__main__":
    start=time.time()
    train()
    print('time',time.time()-start)

