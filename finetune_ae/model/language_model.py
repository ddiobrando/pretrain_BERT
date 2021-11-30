import torch.nn as nn
import torch

from model.bert import BERT

class MLP(torch.nn.Module):
    """
    A multilayer perceptron with ReLU activations and optional BatchNorm.
    Modified from CPA
    """

    def __init__(self, sizes, batch_norm=True, last_layer_act="softmax"):
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
        elif self.activation == "softmax":
            self.softmax = torch.nn.LogSoftmax(dim=-1)
        else:
            raise ValueError("last_layer_act must be one of 'linear' or 'ReLU'")

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        if self.activation == "ReLU":
           x = self.network(x)
           dim = x.size(1) // 2
           return torch.cat((self.relu(x[:, :dim]), x[:, dim:]), dim=1)
        elif self.activation == "softmax":
            return self.softmax(self.network(x))
        return self.network(x)


class BERTLM(nn.Module):
    """
    BERT Language Model
    Classification Model + Masked Language Model
    """

    def __init__(self, bert: BERT, cell_size, drug_size, seq_len):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        self.seq_len = seq_len
        self.cell_prediction = CellPrediction(self.bert.hidden, cell_size)
        self.drug_prediction = DrugPrediction(self.bert.hidden, drug_size)
        self.dose_prediction = DosePrediction(self.bert.hidden)
        self.mask_lm = MaskedLanguageModel(self.bert.hidden, seq_len)
        if bert.n_layers == 0:
            self.mlp = MLP([self.bert.hidden, self.bert.hidden*4, self.bert.hidden, self.bert.hidden*4, self.bert.hidden],last_layer_act="linear")


    def forward(self, x):
        if self.bert.n_layers == 0:
            x = self.bert(x)
            x = self.mlp(x)
        else:
            x = x.reshape((-1,512,self.seq_len))
            x = self.bert(x)
            x = x.reshape((-1,self.bert.hidden))
        return self.cell_prediction(x),self.drug_prediction(x), self.dose_prediction(x), self.mask_lm(x)


class CellPrediction(nn.Module):
    """
    n-class classification model
    """

    def __init__(self, hidden, cell_size):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, cell_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))


class DrugPrediction(nn.Module):
    """
    n-class classification model
    """

    def __init__(self, hidden, drug_size):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        #self.linear0 = nn.Linear(self.bert.hidden, self.bert.hidden)
        #self.tanh = nn.Tanh()
        self.linear = nn.Linear(hidden, drug_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))


class DosePrediction(nn.Module):
    """
    predicting origin token from masked input sequence
    regression problem
    """

    def __init__(self, hidden):
        """
        :param hidden: output size of BERT model
        """
        super().__init__()
        self.linear = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.linear(x)


class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    regression problem
    """

    def __init__(self, hidden, seq_len):
        """
        :param hidden: output size of BERT model
        """
        super().__init__()
        self.linear = nn.Linear(hidden, seq_len)

    def forward(self, x):
        return self.linear(x)
