import torch
import torch.nn as nn
import pdb

from model.bert import BERT

'''#LayerModel
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
        return self.network(x)'''

class BERTLM(nn.Module):
    """
    BERT Language Model
    Classification Model + Masked Language Model
    """

    def __init__(self, bert: BERT, vocab_size, cell_size, drug_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        #kernel_size = 128
        #stride = (seq_len-kernel_size)//10
        #self.conv1d = nn.Conv1d(in_channels=1, out_channels=self.bert.hidden, kernel_size=kernel_size, stride=stride)
        #self.linear1 = MLP([seq_len, self.bert.hidden, self.bert.hidden])
        #self.linear2 = MLP([seq_len, self.bert.hidden, self.bert.hidden])
        self.cell_prediction = CellPrediction(self.bert.hidden, cell_size)
        self.drug_prediction = DrugPrediction(self.bert.hidden, drug_size)
        self.dose_prediction = DosePrediction(self.bert.hidden)
        self.mask_lm = MaskedLanguageModel(self.bert.hidden, vocab_size)

    def forward(self, x):
        """
        :param x: [batch_size, seq_len]
        """
        #x = self.conv1d(x.unsqueeze(1))
        #x = x.permute(0,2,1)
        #x = torch.stack([self.linear1(x),self.linear2(x)],axis=1)
        x = self.bert(x)
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
        return self.softmax(self.linear(x[:, 0, :]))


class DrugPrediction(nn.Module):
    """
    n-class classification model
    """

    def __init__(self, hidden, drug_size):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, drug_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x[:, 1, :]))


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
        return self.linear(x[:, 2, :])


class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    regression problem
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        """
        super().__init__()
        #kernel_size = 128
        #stride = (seq_len-kernel_size)//10
        #self.conv = nn.ConvTranspose1d(in_channels=hidden, out_channels=1, kernel_size=kernel_size, stride=stride)
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        """
        :param: x: [batch_size, seq_len, hidden]
        """
        #x = x.permute(0, 2, 1)
        #x = self.conv(x).squeeze(2)
        #return x
        return self.softmax(self.linear(x))
