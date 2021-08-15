import torch.nn as nn

from model.bert import BERT


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
        self.embedding = nn.Linear(seq_len, self.bert.hidden)
        self.cell_prediction = CellPrediction(self.bert.hidden, cell_size)
        self.drug_prediction = DrugPrediction(self.bert.hidden, drug_size)
        self.dose_prediction = DosePrediction(self.bert.hidden)
        self.mask_lm = MaskedLanguageModel(self.bert.hidden, seq_len)

    def forward(self, x):
        x = self.bert(self.embedding(x))
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
        return self.softmax(self.linear(x)).squeeze(1)


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
        return self.softmax(self.linear(x)).squeeze(1)


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
        return self.linear(x).squeeze(1)


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
