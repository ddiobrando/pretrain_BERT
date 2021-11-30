#%%
import torch
autoencoder = torch.load("pretrain/ae.lm.ep60.pth")
param_group = autoencoder.optimizer_autoencoder.param_groups


# %%
import torch
model = torch.load("output/lr3.novoc.1024.2.2.newae/model_seed=0_epoch=0.pt")
print(model)


# %%
model = torch.load("output/lr3.novoc.1024.2.2.newae/model_seed=0_epoch=1.pt")
print(model)

# %%
