#%%
import random
import torch
from torch import nn
import json
import pandas as pd
from matplotlib import pyplot as plt
from dataset import BERTDataset
from torch.utils.data import DataLoader
import tqdm
import numpy as np
import os
import pdb
from sklearn.metrics import precision_recall_curve,roc_auc_score,auc
os.environ["CUDA_VISIBLE_DEVICES"]="7"

plt.switch_backend('agg')

def write(log_name, post_fix):
    if os.path.exists(log_name):
        with open(log_name, 'r') as f:
            data = json.load(f)
    else:
        data = []
    if type(post_fix)==type([]):
        data.extend(post_fix)
    else:
        data.append(post_fix)
    with open(log_name, 'w') as f:
        json.dump(data, f)

#result_dir = "output/lr4_wd4_oneword_1024_2_2/"
#result_dir = "output/lr4_wd4_oneword_1024_0_0/"
result_dir = "output/lr54_sche_nowd_oneword_1024_2_2/"
model_dir = result_dir+"lm.ep199.pth"

with open(model_dir.replace(".pth",".evaluate.log"),"w") as f:
    f.write("[]")

for seed in range(10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


    model=torch.load(model_dir, map_location="cuda:0")
    if seed == 0:
        print(model)

    nll = nn.NLLLoss()
    mse = nn.MSELoss(reduction='mean')
    cell_vocab_path ="/rd1/user/tanyh/perturbation/pretrain_BERT/cell_vocab.txt"
    drug_vocab_path="/rd1/user/tanyh/perturbation/pretrain_BERT/drug_vocab.txt"
    test_dataset="/rd1/user/tanyh/perturbation/dataset/trt_cp_landmarkonly_test.gctx"
    #test_dataset="/rd1/user/tanyh/perturbation/dataset/trt_cp_landmarkonly_train.gctx"
    gene_thre_path="/rd1/user/tanyh/perturbation/pretrain_BERT/dist_info.csv"
    num_workers=5
    batch_size=2048
    device = 'cuda:0'

    with open(cell_vocab_path) as f:
        cell_vocab = f.read().split('\n')

    with open(drug_vocab_path) as f:
        drug_vocab = f.read().split('\n')
        
    test_dataset = BERTDataset(test_dataset, cell_vocab,drug_vocab, 978)
    data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)


    data_iter = tqdm.tqdm(enumerate(data_loader),
                        total=len(data_loader),
                        bar_format="{l_bar}{r_bar}")

    avg_loss = 0.0
    total_correct = np.zeros(2)
    total_element = np.zeros(2)
    total_dose_loss = 0.0
    total_mask_loss = 0.0
    total_dose_element = 0
    total_mask_element = 0

    for i, data in data_iter:
        model.eval()
        torch.set_grad_enabled(False)

        # 0. batch_data will be sent into the device(GPU or cpu)
        data = {key: value.to(device) for key, value in data.items()}

        # 1. forward the next_sentence_prediction and masked_lm model
        cell_output,drug_output, dose_output, mask_lm_output = model.forward(data["bert_input"])

        cell_loss = nll(cell_output, data["cell"])
        drug_loss = nll(drug_output, data["drug"])
        dose_loss = mse(dose_output, data["dose"])    

        # 2-2. MSELoss of predicting masked token word
        mask=data["bert_label"]>-66
        mask_loss = mse(mask_lm_output[mask], data["bert_label"][mask])

        # 2-3. Adding class_loss and mask_loss : 3.4 Pre-training Procedure
        loss = cell_loss + drug_loss + dose_loss + mask_loss


        avg_loss += loss.item()

        label_nelement = data["bert_label"][mask].nelement()
        total_mask_loss += mask_loss.item()*label_nelement
        total_mask_element += label_nelement
        # prediction accuracy
        for k,(output, class_type) in enumerate(zip([cell_output, drug_output], ["cell", "drug"])):
            correct = output.argmax(dim=-1).eq(data[class_type]).sum().item()
            total_correct[k] += correct
            total_element[k] += data[class_type].nelement()
        total_dose_loss += dose_loss.item()*data["dose"].nelement()
        total_dose_element += data["dose"].nelement()


    epoch_log = {
    "avg_loss":avg_loss / len(data_iter),
    "total_acc":(total_correct * 100.0 / total_element).tolist(),
    "total_mask_rmse":(total_mask_loss/total_mask_element)**0.5,
    "total_dose_rmse":(total_dose_loss/total_dose_element)**0.5,
    }

    print(epoch_log)
    write(model_dir.replace(".pth",".evaluate.log"),epoch_log)

# %%
from scipy.stats import ttest_rel
model1_dir = "output/lr4_wd4_oneword_1024_2_2/lm.ep199.evaluate.log"
with open(model1_dir, 'r') as f:
    model1_eval = json.load(f)
cell1 = []
drug1 = []
dose1 = []
mask1 = []
for e in model1_eval:
    cell1.append(e['total_acc'][0])
    drug1.append(e['total_acc'][1])
    dose1.append(e['total_dose_rmse'])
    mask1.append(e['total_mask_rmse'])

model2_dir = "output/lr54_sche_nowd_oneword_1024_2_2/lm.ep199.evaluate.log"
with open(model2_dir, 'r') as f:
    model2_eval = json.load(f)
cell2 = []
drug2 = []
dose2 = []
mask2 = []
for e in model2_eval:
    cell2.append(e['total_acc'][0])
    drug2.append(e['total_acc'][1])
    dose2.append(e['total_dose_rmse'])
    mask2.append(e['total_mask_rmse'])
print(len(model1_eval),len(model2_eval))

cell_ttest = ttest_rel(cell1, cell2)
drug_ttest = ttest_rel(drug1, drug2)
dose_ttest = ttest_rel(dose1, dose2)
mask_ttest = ttest_rel(mask1, mask2)
print(cell_ttest)
print(drug_ttest)
print(dose_ttest)
print(mask_ttest)
# %%
