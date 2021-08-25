#%%
import json
import pandas as pd
from matplotlib import pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"]="7"

plt.switch_backend('agg')

result_dir = "/rd1/user/tanyh/perturbation/pretrain_BERT_splittask/output/split_3070_128_2_2/"

#%%
mode = 'train'
print("evaluate " + mode)
with open(result_dir + mode + "_log.json") as f:
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
plt.title(mode+'_avg_loss')
plt.savefig(result_dir + mode + "_avg_loss.png")
plt.close()
plt.plot(loss)
plt.title(mode+'loss')
plt.savefig(result_dir + mode + "_loss.png")
plt.close()

plt.plot(cell_loss)
plt.title(mode+'cell_loss')
plt.savefig(result_dir + mode + "_cell_loss.png")
plt.close()

plt.plot(drug_loss)
plt.title(mode+'drug_loss')
plt.savefig(result_dir + mode + "_drug_loss.png")
plt.close()

plt.plot(dose_loss)
plt.title(mode+'dose_loss')
plt.savefig(result_dir + mode + "_dose_loss.png")
plt.close()

plt.plot(mask_loss)
plt.title(mode+'mask_loss')
plt.savefig(result_dir + mode + "_mask_loss.png")
plt.close()

with open(result_dir + mode + "_epoch.json") as f:
    result = json.load(f)
cell_acc = []
drug_acc = []
dose_loss = []
mask_loss = []
avg_loss = []
mask_acc = []
HAVE_MASK_ACC=False
if len(result[0]["total_acc"])==3:
    HAVE_MASK_ACC=True

for line in result:
    cell_acc.append(line["total_acc"][0])
    drug_acc.append(line["total_acc"][1])
    dose_loss.append(line["total_dose_rmse"])
    #mask_loss.append(line["total_mask_rmse"])
    avg_loss.append(line["avg_loss"])
    if HAVE_MASK_ACC:
        mask_acc.append(line["total_acc"][2])
train = pd.DataFrame({
    "cell_acc": cell_acc,
    "drug_acc": drug_acc,
    "dose_rmse": dose_loss,
    "avg_loss": avg_loss
    #"mask_rmse": mask_loss
})
if HAVE_MASK_ACC:
    train.loc[:,"mask_acc"]=mask_acc
train.to_csv(result_dir+mode+"_epoch.tsv",sep="\t")
print(train)

mode = 'test'
print("evaluate " + mode)
with open(result_dir + mode + "_log.json") as f:
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
plt.title(mode+'_avg_loss')
plt.savefig(result_dir + mode + "_avg_loss.png")
plt.close()
plt.plot(loss)
plt.title(mode+'loss')
plt.savefig(result_dir + mode + "_loss.png")
plt.close()

plt.plot(cell_loss)
plt.title(mode+'cell_loss')
plt.savefig(result_dir + mode + "_cell_loss.png")
plt.close()

plt.plot(drug_loss)
plt.title(mode+'drug_loss')
plt.savefig(result_dir + mode + "_drug_loss.png")
plt.close()

plt.plot(dose_loss)
plt.title(mode+'dose_loss')
plt.savefig(result_dir + mode + "_dose_loss.png")
plt.close()

plt.plot(mask_loss)
plt.title(mode+'mask_loss')
plt.savefig(result_dir + mode + "_mask_loss.png")
plt.close()

with open(result_dir + mode + "_epoch.json") as f:
    result = json.load(f)
cell_acc = []
drug_acc = []
dose_loss = []
mask_loss = []
avg_loss = []
mask_acc = []
HAVE_MASK_ACC=False
if len(result[0]["total_acc"])==3:
    HAVE_MASK_ACC=True

for line in result:
    cell_acc.append(line["total_acc"][0])
    drug_acc.append(line["total_acc"][1])
    dose_loss.append(line["total_dose_rmse"])
    #mask_loss.append(line["total_mask_rmse"])
    avg_loss.append(line["avg_loss"])
    if HAVE_MASK_ACC:
        mask_acc.append(line["total_acc"][2])

test = pd.DataFrame({
    "cell_acc": cell_acc,
    "drug_acc": drug_acc,
    "dose_rmse": dose_loss,
    "avg_loss": avg_loss
    #"mask_rmse": mask_loss
})
if HAVE_MASK_ACC:
    test.loc[:,"mask_acc"]=mask_acc

test.to_csv(result_dir+mode+"_epoch.tsv",sep="\t")
print(test)
#%%

