#%%
import json
import pandas as pd
from matplotlib import pyplot as plt
plt.switch_backend('agg')

result_dir = "../output/08151149/"
mode = 'train'
print("evaluate "+mode)
with open(result_dir+mode+"_log.json") as f:
    result = json.load(f)
avg_loss = []
loss = []
cell_acc = []
drug_acc = []
dose_loss = []
mask_loss = []
for line in result:
    avg_loss.append(line['avg_loss'])
    loss.append(line["loss"])
plt.plot(avg_loss)
plt.title('train_avg_loss')
plt.savefig(result_dir+mode+"_avg_loss.png")
plt.close()
plt.plot(loss)
plt.title('loss')
plt.savefig(result_dir+mode+"_loss.png")
plt.close()

with open(result_dir+mode+"_epoch.json") as f:
    result = json.load(f)
cell_acc = []
drug_acc = []
dose_loss = []
mask_loss = []
for line in result:
    cell_acc.append(line["total_acc"][0])
    drug_acc.append(line["total_acc"][1])
    dose_loss.append(line["dose_loss"])
    mask_loss.append(line["mask_loss"])
train = pd.DataFrame({"cell_acc":cell_acc,"drug_acc":drug_acc,"dose_loss":dose_loss,"mask_loss":mask_loss})
display(train)


mode = 'test'
print("evaluate "+mode)
with open(result_dir+mode+"_epoch.json") as f:
    result = json.load(f)
cell_acc = []
drug_acc = []
dose_loss = []
mask_loss = []
for line in result:
    cell_acc.append(line["total_acc"][0])
    drug_acc.append(line["total_acc"][1])
    dose_loss.append(line["dose_loss"])
    mask_loss.append(line["mask_loss"])
test = pd.DataFrame({"cell_acc":cell_acc,"drug_acc":drug_acc,"dose_loss":dose_loss,"mask_loss":mask_loss})
display(test)
#%%
