import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from typing import *
import pdb

class GeneralizedSigmoid(torch.nn.Module):
    """
    Sigmoid, log-sigmoid or linear functions for encoding dose-response for
    drug perurbations.
    """

    def __init__(self, dim, device, nonlin='sigmoid'):
        """Sigmoid modeling of continuous variable.
        Params
        ------
        nonlin : str (default: logsigm)
            One of logsigm, sigm.
        """
        super(GeneralizedSigmoid, self).__init__()
        self.nonlin = nonlin
        self.beta = torch.nn.Parameter(
            torch.ones(1, dim, device=device),
            requires_grad=True
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(1, dim, device=device),
            requires_grad=True
        )

    def forward(self, x):
        if self.nonlin == 'logsigm':
            c0 = self.bias.sigmoid()
            return (torch.log1p(x) * self.beta + self.bias).sigmoid() - c0
        elif self.nonlin == 'sigm':
            c0 = self.bias.sigmoid()
            return (x * self.beta + self.bias).sigmoid() - c0
        else:
            return x

    def one_drug(self, x, i):
        if self.nonlin == 'logsigm':
            c0 = self.bias[0][i].sigmoid()
            return (torch.log1p(x) * self.beta[0][i] + self.bias[0][i]).sigmoid() - c0
        elif self.nonlin == 'sigm':
            c0 = self.bias[0][i].sigmoid()
            return (x * self.beta[0][i] + self.bias[0][i]).sigmoid() - c0
        else:
            return x

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


class AE(nn.Module):
    def __init__(self,
                 num_genes: int,
                 num_classes: int,
                 class_sizes: List[int],
                 device: str = 'cuda:0',
                 seed: int = 0,
                 hparams="") -> None:
        super().__init__()
        
        # set hyperparameters
        self.set_hparams_(seed, hparams)
        self.model = 'AE'
        self.num_genes = num_genes
        num_drugs = class_sizes[0]
        num_cell_types = class_sizes[1]
        self.num_drugs = num_drugs
        self.num_cell_types = num_cell_types
        self.device = device

        # set models
        self.encoder = MLP(
            [num_genes] +
            [self.hparams["autoencoder_width"]] *
            self.hparams["autoencoder_depth"] +
            [self.hparams["dim"]])

        self.decoder = MLP(
            [self.hparams["dim"]] +
            [self.hparams["autoencoder_width"]] *
            self.hparams["autoencoder_depth"] +
            [num_genes], last_layer_act="linear")

        self.drug_embeddings = torch.nn.Embedding(
            num_drugs, self.hparams["dim"])
        self.cell_type_embeddings = torch.nn.Embedding(
            num_cell_types, self.hparams["dim"])

        self.dosers = GeneralizedSigmoid(num_drugs, self.device,
            nonlin="sigm")

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
            "dim": 256 if default else
            int(np.random.choice([128, 256, 512])),
            "autoencoder_width": 512 if default else
            int(np.random.choice([256, 512, 1024])),
            "autoencoder_depth": 4 if default else
            int(np.random.choice([3, 4, 5])),
            "autoencoder_wd": 1e-6 if default else
            float(10**np.random.uniform(-8, -4)),
            "step_size_lr": 25 if default else
            int(np.random.choice([15, 25, 45])),
            "batch_size": 1024,
            "lr": 0.001 if default else
            float(np.random.choice([0.005, 0.001, 0.0005])),
        }
        
        # the user may fix some hparams
        if hparams != "":
            if isinstance(hparams, str):
                self.hparams.update(json.loads(hparams))
            else:
                self.hparams.update(hparams)

        return self.hparams

    def forward(self, y: Tensor, cats: List[Tensor], **kwargs) -> List[Tensor]:
        drug_emb = self.dosers(cats[0]) @ self.drug_embeddings.weight
        cell_emb = self.cell_type_embeddings(cats[1].argmax(1))
        z = self.encoder(y)
        pred_y = self.decoder(z+drug_emb+cell_emb)

        return pred_y


    def move_inputs_(self, y):
        """
        Move minibatch tensors to CPU/GPU.
        """
        if torch.is_tensor(y):
            return y.to(self.device)
        else:
            return [item.to(self.device) for item in y]
    
    def update(self, genes, drugs, cell_types, train=False):
        genes, drugs, cell_types = self.move_inputs_([genes, drugs, cell_types])
        optimizer = self.optimizer_autoencoder
        outputs = self(genes, [drugs, cell_types])
        tr_loss = F.mse_loss(outputs, genes, reduction='mean')
        if train:
            optimizer.zero_grad()
            tr_loss.backward()
            optimizer.step()
        return tr_loss.item()

    def check_for_early_break_condition(self, val_losses, required_window=5):
        if len(val_losses) < required_window:
            return False
        # return True if val_loss has strictly increased within the required window
        windowOI = val_losses[-required_window:]
        return min(windowOI) == windowOI[0]  # has been increasing