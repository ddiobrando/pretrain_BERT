import torch
from torch import nn
from model import BERT

class Imputation(nn.Module):

    def __init__(self, bert: BERT, gene_thre, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        self.special = ["PAD","CLS","MASK","UNK"]
        self.pad_index = 0
        self.cls_index = 1
        self.mask_index = 2
        self.unk_index = 3
        self.gene_thre = gene_thre
        self.seq_len = 512
        self.vocab_size = vocab_size
        self.sel_thre = list(range(5,99,10))
        self.linear1 = nn.Linear(bert.hidden, vocab_size)
        self.linear2 = nn.Linear(self.seq_len, 1)
        self.linear3 = nn.Linear(len(self.sel_thre), 1)


    def forward(self, x, device):
        """
        :param x: [batch_size, seq_len]
        """
        all_tokens = []
        batch_size, _ = x.shape[0], x.shape[1]
        for t1 in x:
            for thre_idx in self.sel_thre:
                t1_tokens = []
                if thre_idx < 50:
                    for gene_idx, thre in enumerate(self.gene_thre.iloc[thre_idx, :].values):
                        if t1[gene_idx] < thre:
                            t1_tokens.append(gene_idx+len(self.special))
                else:
                    for gene_idx, thre in enumerate(self.gene_thre.iloc[thre_idx, :].values):
                        if t1[gene_idx] > thre:
                            t1_tokens.append(gene_idx+len(self.special))
                t1_tokens = [self.cls_index] + t1_tokens
                bert_input = (t1_tokens)[:self.seq_len]

                padding = [self.pad_index for _ in range(self.seq_len - len(bert_input))]
                bert_input.extend(padding)
                all_tokens.append(bert_input)
        x = torch.tensor(all_tokens).to(device)
        x = self.bert(x)
        x = self.linear1(x)
        x = x.permute(0, 2, 1)
        x = self.linear2(x).squeeze(2)
        x = x.reshape((batch_size, len(self.sel_thre), self.vocab_size)).permute(0,2,1)
        x = self.linear3(x).squeeze(2)
        return x