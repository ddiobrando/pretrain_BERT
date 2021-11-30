import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from typing import *
import pdb


class cVAE_Sequence(nn.Module):
    def __init__(self,
                 input_size: int,
                 num_classes: int,
                 class_sizes: List[int],
                 device: str = 'cuda:0',
                 seed: int = 0,
                 hparams="") -> None:
        super().__init__()
        # set hyperparameters
        self.set_hparams_(seed, hparams)
        self.model = 'cvae'
        self.input_size = input_size
        self.num_classes = num_classes
        self.class_sizes = class_sizes
        self.latent_dim = self.hparams['latent_dim']
        self.device = device
        width = self.hparams['autoencoder_width']
        depth = self.hparams['autoencoder_depth']
        hidden_dims =[width for i in range(depth)]
        self.hidden_dims = hidden_dims  # need this for ODE interaction!
        
        prev_d = input_size + np.sum(class_sizes)
        enc_layers = []
        for i in range(len(hidden_dims)):
            enc_layers.append(nn.Sequential(nn.Linear(prev_d, hidden_dims[i]), nn.ReLU()))
            prev_d = hidden_dims[i]
        latent_dim = self.latent_dim
        self.encoder = nn.Sequential(*enc_layers)
        self.mu_fc = nn.Linear(prev_d, latent_dim)
        self.var_fc = nn.Linear(prev_d, latent_dim)

        prev_d = latent_dim + np.sum(class_sizes)
        dec_layers = []
        for i in range(len(hidden_dims)-1, -1, -1):  # go backwards through hidden dims
            dec_layers.append(nn.Sequential(nn.Linear(prev_d, hidden_dims[i]), nn.ReLU()))
            prev_d = hidden_dims[i]
        dec_layers.append(nn.Sequential(nn.Linear(prev_d, input_size), nn.ReLU()))  # output is 1-d
        self.decoder = nn.Sequential(*dec_layers)
        
        # optimizers
        self.optimizer_autoencoder = torch.optim.Adam(
            self.parameters(),lr=self.hparams["lr"],weight_decay=self.hparams["autoencoder_wd"])
            
        # learning rate schedulers
        self.scheduler_autoencoder = torch.optim.lr_scheduler.StepLR(
            self.optimizer_autoencoder, step_size=self.hparams["step_size_lr"])

        self.history = {'epoch': [], 'stats_epoch': []}

    def set_hparams_(self, seed, hparams):
        """
        Set hyper-parameters to (i) default values if `seed=0`, (ii) random
        values if `seed != 0`, or (iii) values fixed by user for those
        hyper-parameters specified in the JSON string `hparams`.
        """

        default = (seed == 0)
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        self.hparams = {
            "latent_dim": 256 if default else
            int(np.random.choice([128, 256, 512])),
            "autoencoder_width": 512 if default else
            int(np.random.choice([256, 512, 1024])),
            "autoencoder_depth": 4 if default else
            int(np.random.choice([3, 4, 5])),
            "autoencoder_wd": 1e-6 if default else #weight decay
            float(10**np.random.uniform(-8, -4)),
            "step_size_lr": 45 if default else
            int(np.random.choice([15, 25, 45])),
            "batch_size": 1024,
            "lr": 1e-3 if default else
            float(np.random.choice([0.005, 0.001, 0.0005])),
        }
        
        # the user may fix some hparams
        if hparams != "":
            if isinstance(hparams, str):
                self.hparams.update(json.loads(hparams))
            else:
                self.hparams.update(hparams)

        return self.hparams


    def encode(self, x: Tensor) -> List[Tensor]:
        x = self.encoder(x)
        return self.mu_fc(x), self.var_fc(x)

    def decode(self, x: Tensor) -> Tensor:
        return self.decoder(x)

    def reparameterize(self, mu: Tensor, log_var: Tensor, **kwargs) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        if 'return_eps' in kwargs and kwargs['return_eps']:
            return eps * std + mu, eps
        return eps * std + mu

    def forward(self, y: Tensor, cats: List[Tensor], **kwargs) -> List[Tensor]:
        y = self.move_inputs_(y)
        cats = self.move_inputs_(cats)
        x = torch.cat([y, *cats], dim=1)
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var, **kwargs)
        z = torch.cat([z, *cats], dim=1)
        pred_y = self.decode(z)
        
        return pred_y, y, mu, log_var

    def loss_fn(self, *args, **kwargs) -> dict:
        pred_y = args[0]
        y = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['beta']

        recons_y_loss = F.mse_loss(pred_y, y, reduction='mean')
        kld_loss = torch.mean(-0.5 * torch.sum(
            1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_y_loss + kld_weight * kld_loss
        return {'loss': loss, 'MSE': recons_y_loss, 'KLD': -kld_loss}

    def sample(self, num_samples: int, current_device: int, cats: List[Tensor], **kwargs) -> Tensor:
        z = torch.rand(size=(num_samples, self.latent_dim))
        z = torch.cat([z, *cats])
        return self.decode(z)

    def generate(self, y: Tensor, old_cats: List[Tensor], new_cats: List[Tensor], **kwargs) -> Tensor:
        y = self.move_inputs_(y)
        old_cats = self.move_inputs_(old_cats)
        new_cats = self.move_inputs_(new_cats)
        x = torch.cat([y, *old_cats], dim=1)
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        z = torch.cat([z, *new_cats], dim=1)
        return self.decode(z)

    def get_latent_representation(self, y: Tensor, cats: List[Tensor], **kwargs) -> Tensor:
        x = torch.cat([y, *cats], dim=1)
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return z

    def move_inputs_(self, y):
        """
        Move minibatch tensors to CPU/GPU.
        """
        if torch.is_tensor(y):
            return y.to(self.device)
        else:
            return [item.to(self.device) for item in y]
    
    def update(self, genes, drugs, cell_types):
        optimizer = self.optimizer_autoencoder
        outputs = self(genes, [drugs, cell_types])
        tr_loss = self.loss_fn(*outputs, beta=1.0)
        optimizer.zero_grad()
        tr_loss['loss'].backward()
        optimizer.step()
        return {
            "tr_loss": tr_loss['loss'].item(),
        }

    def check_for_early_break_condition(self, val_losses, required_window=5):
        if len(val_losses) < required_window:
            return False
        # return True if val_loss has strictly increased within the required window
        windowOI = val_losses[-required_window:]
        return min(windowOI) == windowOI[0]  # has been increasing