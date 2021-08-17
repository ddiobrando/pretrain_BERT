#%%
import torch
import json
import pandas as pd
from matplotlib import pyplot as plt
from dataset import BERTDataset
from torch.utils.data import DataLoader
import tqdm
import numpy as np


plt.switch_backend('agg')

#result_dir = "/rd1/user/tanyh/perturbation/pretrain_BERT/output/0816basemlp1e-4bs128/"
#result_dir = "/rd1/user/tanyh/perturbation/pretrain_BERT/output/0816basemlp/"
result_dir = "/rd1/user/tanyh/perturbation/pretrain_BERT/output/0817/"

#%%
mode = 'train'
print("evaluate " + mode)
"""with open(result_dir + mode + "_log.json") as f:
    result = json.load(f)
avg_loss = []
loss = []
cell_acc = []
drug_acc = []
cell_loss = []
drug_loss = []
dose_loss = []
mask_loss = []
for line in result:
    avg_loss.append(line['avg_loss'])
    loss.append(line["loss"])
    cell_loss.append(line["cell_loss"])
    drug_loss.append(line["drug_loss"])
    dose_loss.append(line["dose_loss"])
    mask_loss.append(line["mask_loss"])
plt.plot(avg_loss)
plt.title('train_avg_loss')
plt.savefig(result_dir + mode + "_avg_loss.png")
plt.close()
plt.plot(loss)
plt.title('loss')
plt.savefig(result_dir + mode + "_loss.png")
plt.close()

plt.plot(cell_loss)
plt.title('cell_loss')
plt.savefig(result_dir + mode + "_cell_loss.png")
plt.close()

plt.plot(drug_loss)
plt.title('drug_loss')
plt.savefig(result_dir + mode + "_drug_loss.png")
plt.close()

plt.plot(dose_loss)
plt.title('dose_loss')
plt.savefig(result_dir + mode + "_dose_loss.png")
plt.close()

plt.plot(mask_loss)
plt.title('mask_loss')
plt.savefig(result_dir + mode + "_mask_loss.png")
plt.close()"""

with open(result_dir + mode + "_epoch.json") as f:
    result = json.load(f)
cell_acc = []
drug_acc = []
dose_loss = []
mask_loss = []
for line in result:
    cell_acc.append(line["total_acc"][0])
    drug_acc.append(line["total_acc"][1])
    dose_loss.append(line["total_dose_rmse"])
    mask_loss.append(line["total_mask_rmse"])
train = pd.DataFrame({
    "cell_acc": cell_acc,
    "drug_acc": drug_acc,
    "dose_rmse": dose_loss,
    "mask_rmse": mask_loss
})
display(train)

mode = 'test'
print("evaluate " + mode)
with open(result_dir + mode + "_epoch.json") as f:
    result = json.load(f)
cell_acc = []
drug_acc = []
dose_loss = []
mask_loss = []
for line in result:
    cell_acc.append(line["total_acc"][0])
    drug_acc.append(line["total_acc"][1])
    dose_loss.append(line["total_dose_rmse"])
    mask_loss.append(line["total_mask_rmse"])
test = pd.DataFrame({
    "cell_acc": cell_acc,
    "drug_acc": drug_acc,
    "dose_rmse": dose_loss,
    "mask_rmse": mask_loss
})
test.to_csv(result_dir+mode+"_epoch.tsv",sep="\t")
display(test)
#%%
model=torch.load(result_dir+"lm.ep5.pth")

cell_vocab_path ="/rd1/user/tanyh/perturbation/pretrain_BERT/cell_vocab.txt"
drug_vocab_path="/rd1/user/tanyh/perturbation/pretrain_BERT/drug_vocab.txt"
test_dataset="/rd1/user/tanyh/perturbation/pretrain_BERT/trt_cp_landmarkonly_test.gctx"
num_workers=5
batch_size=128
device = 'cpu'

with open(cell_vocab_path) as f:
    cell_vocab = f.read().split('\n')

with open(drug_vocab_path) as f:
    drug_vocab = f.read().split('\n')

test_dataset = BERTDataset(test_dataset, cell_vocab,drug_vocab)
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

mse = torch.nn.MSELoss(reduction='mean')

for i, data in data_iter:
    # 0. batch_data will be sent into the device(GPU or cpu)
    data = {key: value.to(device) for key, value in data.items()}

    # 1. forward the next_sentence_prediction and masked_lm model
    cell_output,drug_output, dose_output, mask_lm_output = model.forward(data["bert_input"])

    dose_loss = mse(dose_output, data["dose"])
    mask_loss = mse(mask_lm_output, data["bert_label"])
    print('pred', mask_lm_output)
    print('true', data["bert_label"])
    break

    # prediction accuracy
    for k,(output, class_type) in enumerate(zip([cell_output, drug_output], ["cell", "drug"])):
        correct = output.argmax(dim=-1).eq(data[class_type]).sum().item()
        total_correct[k] += correct
        total_element[k] += data[class_type].nelement()
    total_dose_loss += dose_loss.item()*data["dose"].nelement()
    total_dose_element += data["dose"].nelement()
    total_mask_loss += mask_loss.item()*data["bert_input"].nelement()
    total_mask_element += data["bert_input"].nelement()

epoch_log = {"avg_loss":avg_loss / len(data_iter),
"total_acc":(total_correct * 100.0 / total_element).tolist(),
"total_dose_rmse":(total_dose_loss/total_dose_element)**0.5,
"total_mask_rmse":(total_mask_loss/total_mask_element)**0.5}
print(epoch_log)

# %%
