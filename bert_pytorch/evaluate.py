#%%
import torch
import json
import pandas as pd
from matplotlib import pyplot as plt
from dataset import BERTDataset
from torch.utils.data import DataLoader
import tqdm
import numpy as np
import os
import pdb
os.environ["CUDA_VISIBLE_DEVICES"]="4"

plt.switch_backend('agg')

result_dir = "/rd1/user/tanyh/perturbation/pretrain_BERT/output/0821_3070_128_2_2/"
#result_dir = "/rd1/user/tanyh/perturbation/pretrain_BERT/output/0821_3070_128_0_0/"

model=torch.load(result_dir+"lm.ep49.pth", map_location="cuda:0")
cell_vocab_path ="/rd1/user/tanyh/perturbation/pretrain_BERT/cell_vocab.txt"
drug_vocab_path="/rd1/user/tanyh/perturbation/pretrain_BERT/drug_vocab.txt"
test_dataset="/rd1/user/tanyh/perturbation/dataset/trt_cp_landmarkonly_test.gctx"
#test_dataset="/rd1/user/tanyh/perturbation/dataset/trt_cp_landmarkonly_train.gctx"
gene_thre_path="/rd1/user/tanyh/perturbation/pretrain_BERT/dist_info.csv"
num_workers=5
batch_size=64
device = 'cuda:0'

with open(cell_vocab_path) as f:
    cell_vocab = f.read().split('\n')

with open(drug_vocab_path) as f:
    drug_vocab = f.read().split('\n')

gene_thre = pd.read_csv(gene_thre_path, index_col=0)

test_dataset = BERTDataset(test_dataset, cell_vocab,drug_vocab, gene_thre)
data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)


data_iter = tqdm.tqdm(enumerate(data_loader),
                        total=len(data_loader),
                        bar_format="{l_bar}{r_bar}")

avg_loss = 0.0
total_correct = np.zeros(3)
total_element = np.zeros(3)
total_dose_loss = 0.0
total_mask_loss = 0.0
total_dose_element = 0
total_mask_element = 0

mse = torch.nn.MSELoss(reduction='mean')

for i, data in data_iter:
    model.eval()
    torch.set_grad_enabled(False)
    # 0. batch_data will be sent into the device(GPU or cpu)
    data = {key: value.to(device) for key, value in data.items()}

    # 1. forward the next_sentence_prediction and masked_lm model
    cell_output,drug_output, dose_output, mask_lm_output = model.forward(data["bert_input"])

    dose_loss = mse(dose_output, data["dose"])

    #mask_loss = mse(mask_lm_output, data["bert_label"])
    #print('pred', mask_lm_output)
    #print('true', data["bert_label"])

    # prediction accuracy
    for k,(output, class_type) in enumerate(zip([cell_output, drug_output], ["cell", "drug"])):
        correct = output.argmax(dim=-1).eq(data[class_type]).sum().item()
        total_correct[k] += correct
        total_element[k] += data[class_type].nelement()
    valid_idx = data["bert_label"]!=0
    batch_size, seq_len, vocab_size = mask_lm_output.shape
    masked_mask_output = torch.masked_select(mask_lm_output,valid_idx.unsqueeze(2).expand(batch_size, seq_len, vocab_size)).reshape(-1,vocab_size)
    masked_label = torch.masked_select(data["bert_label"],valid_idx)
    correct = masked_mask_output.argmax(dim=-1).eq(masked_label).sum().item()
    total_correct[2] += correct
    total_element[2] += masked_label.nelement()
    
    total_dose_loss += dose_loss.item()*data["dose"].nelement()
    total_dose_element += data["dose"].nelement()

epoch_log = {"avg_loss":avg_loss / len(data_iter),
"total_acc":(total_correct * 100.0 / total_element).tolist(),
"total_dose_rmse":(total_dose_loss/total_dose_element)**0.5}
print(epoch_log)


# %%
