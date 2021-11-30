import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from typing import *
from compert.model import MLP, GeneralizedSigmoid
import pdb
from model.bert import BERT


class PreAE(nn.Module):
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
        self.model = 'BERT'
        self.input_size = input_size
        self.num_classes = num_classes
        self.class_sizes = class_sizes
        num_drugs = class_sizes[0]
        num_cell_types = class_sizes[1]
        self.device = device
        #width = self.hparams['autoencoder_width']
        #depth = self.hparams['autoencoder_depth']
        
        def build_encoder(encoder_dims, prev_d):
            layers = []
            for i in range(len(encoder_dims)):
                layers.append(nn.Sequential(nn.Linear(prev_d, encoder_dims[i]), nn.ReLU()))
                prev_d = encoder_dims[i]
            return nn.Sequential(*layers)

        # bert encoder
        #self.encoder = torch.load("pretrain/lr54.wd0.1024.2.2.bert.ep199.pth")
        #self.encoder = torch.load("pretrain/lr4.128.2.2.bert.ep108.pth")
        self.encoder = torch.load("pretrain/mask80.bert.ep200.pth")
        #self.encoder = BERT(hidden=512,n_layers=2,attn_heads=2,seq_len=5000)
        latent_dim = self.encoder.hidden

        # Build expr encoder
        expr_encoder_dims = [1024,latent_dim]
        prev_d = input_size
        self.expr_encoder = build_encoder(expr_encoder_dims,prev_d)

        self.drug_embeddings = torch.nn.Embedding(
            num_drugs, latent_dim)
        self.cell_type_embeddings = torch.nn.Embedding(
            num_cell_types, latent_dim)
        #self.linear=nn.Linear(input_size,latent_dim)

        #self.mu_fc = nn.Linear(latent_dim+hidden_dims[-1], latent_dim)
        #self.var_fc = nn.Linear(latent_dim+hidden_dims[-1], latent_dim)
        #prev_d = latent_dim+np.sum(class_sizes)

        # Build decoder
        decoder_dims = [1024,1024]+expr_encoder_dims
        prev_d = latent_dim
        dec_layers = []
        for i in range(len(decoder_dims)-1, -1, -1):
            dec_layers.append(nn.Sequential(nn.Linear(prev_d, decoder_dims[i]), nn.ReLU()))
            prev_d = decoder_dims[i]
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
            #"latent_dim": 256 if default else
            #int(np.random.choice([128, 256, 512])),
            #"autoencoder_width": 512 if default else
            #int(np.random.choice([256, 512, 1024])),
            #"autoencoder_depth": 4 if default else
            #int(np.random.choice([3, 4, 5])),
            "autoencoder_wd": 1e-8 if default else #weight decay
            float(10**np.random.uniform(-8, -4)),
            "step_size_lr": 25 if default else
            int(np.random.choice([15, 25, 45])),
            "batch_size": 1024,
            "lr": 5e-4 if default else
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
        x = self.expr_encoder(x)
        x = x.unsqueeze(1)
        x = self.encoder(x).squeeze(1)
        return x

    def decode(self, x: Tensor) -> Tensor:
        return self.decoder(x)

    #def reparameterize(self, mu: Tensor, log_var: Tensor, **kwargs) -> Tensor:
    #    std = torch.exp(0.5 * log_var)
    #    eps = torch.randn_like(std)
    #    if 'return_eps' in kwargs and kwargs['return_eps']:
    #        return eps * std + mu, eps
    #    return eps * std + mu

    def forward(self, y: Tensor, cats: List[Tensor], **kwargs) -> List[Tensor]:
        
        drug_emb = self.drug_embeddings(cats[0].argmax(1))
        cell_emb = self.cell_type_embeddings(cats[1].argmax(1))
        z = self.encode(y)
        pred_y = self.decode(z+drug_emb+cell_emb)
        #z = self.encode(self.linear(y.unsqueeze(1)),cats)
        #mu, log_var = self.encode(self.linear(y.unsqueeze(0)),cats)

        #z = self.reparameterize(mu, log_var, **kwargs)

        #z = torch.cat([z, cats],dim=1)
        #pred_y = self.decode(z)

        return pred_y#, y#, mu, log_var

    """def loss_fn(self, *args, **kwargs) -> dict:
        pred_y = args[0]
        y = args[1]
        #mu = args[2]
        #log_var = args[3]

        #kld_weight = kwargs['beta']

        recons_y_loss = F.mse_loss(pred_y, y, reduction='mean')
        #kld_loss = torch.mean(-0.5 * torch.sum(
        #    1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_y_loss #+ kld_weight * kld_loss
        return {'loss': loss, 'MSE': recons_y_loss}#, 'KLD': -kld_loss}"""


    def move_inputs_(self, y):
        """
        Move minibatch tensors to CPU/GPU.
        """
        if torch.is_tensor(y):
            return y.to(self.device)
        else:
            return [item.to(self.device) for item in y]
    
    def update(self, genes, drugs, cell_types):
        genes, drugs, cell_types = self.move_inputs_([genes, drugs, cell_types])
        optimizer = self.optimizer_autoencoder
        outputs = self(genes, [drugs, cell_types])
        tr_loss = F.mse_loss(outputs, genes, reduction='mean')
        #tr_loss = self.loss_fn(*outputs, beta=1.0)
        optimizer.zero_grad()
        tr_loss.backward()
        optimizer.step()
        return {
            "tr_loss": tr_loss.item(),
        }

    def check_for_early_break_condition(self, val_losses, required_window=5):
        if len(val_losses) < required_window:
            return False
        # return True if val_loss has strictly increased within the required window
        windowOI = val_losses[-required_window:]
        return min(windowOI) == windowOI[0]  # has been increasing