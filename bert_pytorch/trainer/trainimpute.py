import tqdm
import torch
import pdb

from torch.utils.data import DataLoader
import pandas as pd
from torch.optim import Adam
from torch import nn
from trainer.optim_schedule import ScheduledOptim
from model.imputation import Imputation

from model import BERT
import json
import os

def write(log_name, post_fix):
    if os.path.exists(log_name):
        with open(log_name, 'r') as f:
            data = json.load(f)
    else:
        data = []
    data.append(post_fix)
    with open(log_name, 'w') as f:
        json.dump(data, f)

class ImputeTrainer:
    """
    """

    def __init__(self, bert: BERT, gene_thre: pd.DataFrame, seq_len: int,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10, log_name="train_info.json",
                 resume=None, optim=None):
        """
        :param bert: BERT model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        # This BERT model will be saved every epoch
        self.bert = bert

        if resume is None:
            # Initialize the BERT Language Model, with BERT model
            self.model = Imputation(bert,gene_thre,gene_thre.shape[1]).to(self.device)
        else:
            self.model = resume.to(self.device)
        print(self.model)
        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        if optim is None:
            # Setting the Adam optimizer with hyper-param
            self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        else:
            self.optim = optim
        self.optim_schedule = ScheduledOptim(self.optim, self.bert.hidden, n_warmup_steps=warmup_steps)

        self.mse = nn.MSELoss(reduction='mean')

        self.log_freq = log_freq
        self.log_name = log_name

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        str_code = "train" if train else "test"
        log_name = self.log_name+str_code+"_log.json"
        epoch_log_name = self.log_name+str_code+"_epoch.json"

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        total_mask_loss = 0.0
        total_mask_element = 0

        for i, data in data_iter:
            if train:
                self.model.train()
                torch.set_grad_enabled(True)
            else:
                self.model.eval()
                torch.set_grad_enabled(False)
            # 0. batch_data will be sent into the device(GPU or cpu)
            #data = {key: value.to(self.device) for key, value in data.items()}
            data["bert_label"] = data["bert_label"].to(self.device)

            # 1. forward the next_sentence_prediction and masked_lm model
            mask_lm_output = self.model.forward(data["bert_input"], self.device)

            # 2-2. MSELoss of predicting masked token word
            mask=data["bert_label"]>-66
            loss = self.mse(mask_lm_output[mask], data["bert_label"][mask])

            # 3. backward and optimization only in train
            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

            avg_loss += loss.item()
            label_nelement = data["bert_label"][mask].nelement()
            total_mask_loss += loss.item()*label_nelement
            total_mask_element += label_nelement
            if i % self.log_freq == 0:
                post_fix = {
                    "epoch": epoch,
                    "iter": i,
                    "avg_loss": avg_loss / (i + 1),
                    "loss": loss.item()
                }
                write(log_name, post_fix)
                data_iter.write(str(post_fix))

        epoch_log = {"epoch":epoch,
        "avg_loss":avg_loss / len(data_iter),
        "total_mask_rmse":(total_mask_loss/total_mask_element)**0.5
        }
        write(epoch_log_name,epoch_log)
        print(epoch_log)

    def save(self, epoch, file_path="output/bert_trained.model"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        bert_output_path = file_path + "bert.ep%d.pth" % epoch
        torch.save(self.bert.cpu(), bert_output_path)
        self.bert.to(self.device)
        lm_output_path = file_path + "impute.ep%d.pth" % epoch
        torch.save(self.optim.state_dict(), lm_output_path.replace("impute","optim"))
        torch.save(self.model.cpu(), lm_output_path)
        self.model.to(self.device)
        print("EP:%d Model Saved on:" % epoch, bert_output_path, lm_output_path)
        return bert_output_path
