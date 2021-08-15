import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
import json
from model import BERTLM, BERT
from trainer.optim_schedule import ScheduledOptim

import tqdm


def write(log_name, post_fix):
    if os.path.exists(log_name):
        with open(log_name, 'r') as f:
            data = json.load(f)
    else:
        data = []
    data.append(post_fix)
    with open(log_name, 'w') as f:
        json.dump(data, f)


class BERTTrainer:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction

    please check the details on README.md with simple example.

    """

    def __init__(self, bert: BERT, cell_vocab_size: int, drug_vocab_size: int, seq_len: int,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10, log_name="train_info.json"):
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
        # Initialize the BERT Language Model, with BERT model
        self.model = BERTLM(bert, cell_vocab_size, drug_vocab_size, seq_len).to(self.device)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.bert.hidden, n_warmup_steps=warmup_steps)

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.nll = nn.NLLLoss(ignore_index=0)
        # Maybe sum will work
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
        total_correct = np.zeros(2)
        total_element = np.zeros(2)
        total_dose_loss = 0.0
        total_mask_loss = 0.0
        total_dose_element = 0
        total_mask_element = 0

        for i, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            # 1. forward the next_sentence_prediction and masked_lm model
            cell_output,drug_output, dose_output, mask_lm_output = self.model.forward(data["bert_input"])

            # 2-1. NLL(negative log likelihood) loss of is_next classification result
            #print('cell_output',cell_output.shape,'cell',data["cell"].shape)
            #print('drug_output',drug_output.shape,'drug',data["drug"].shape)
            #print('dose_output',dose_output.shape,'dose',data["dose"].shape)
            #print('mask',mask_lm_output.shape,'dose',data["bert_label"].shape)

            cell_loss = self.nll(cell_output, data["cell"])
            drug_loss = self.nll(drug_output, data["drug"])
            dose_loss = self.mse(dose_output, data["dose"])

            # 2-2. MSELoss of predicting masked token word
            mask_loss = self.mse(mask_lm_output, data["bert_label"])

            # 2-3. Adding class_loss and mask_loss : 3.4 Pre-training Procedure
            loss = cell_loss + drug_loss + dose_loss + mask_loss
            #print('cell_loss',cell_loss.item(),'drug_loss',drug_loss.item(),'dose_loss',dose_loss.item(),'mask_loss',mask_loss.item())

            # 3. backward and optimization only in train
            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

            avg_loss += loss.item()

            if i % self.log_freq == 0:
                # prediction accuracy
                for k,(output, class_type) in enumerate(zip([cell_output, drug_output], ["cell", "drug"])):
                    correct = output.argmax(dim=-1).eq(data[class_type]).sum().item()
                    total_correct[k] += correct
                    total_element[k] += data[class_type].nelement()
                total_dose_loss += dose_loss*data["dose"].nelement()
                total_dose_element += data["dose"].nelement()
                total_mask_loss += mask_loss*data["bert_input"].nelement()
                total_mask_element += data["bert_input"].nelement()

                post_fix = {
                    "epoch": epoch,
                    "iter": i,
                    "avg_loss": avg_loss / (i + 1),
                    "avg_acc": (total_correct / total_element * 100).tolist(),
                    "cell_loss": cell_loss.item(),
                    "drug_loss": drug_loss.item(),
                    "dose_loss": dose_loss.item(),
                    "mask_loss": mask_loss.item(),
                    "loss": loss.item()
                }
                write(log_name, post_fix)
                data_iter.write(str(post_fix))


        epoch_log = {"epoch":epoch,
        "avg_loss":avg_loss / len(data_iter),
        "total_acc":(total_correct * 100.0 / total_element).tolist(),
        "total_dose_rmse":(total_dose_loss/total_dose_element)**0.5,
        "total_mask_rmse":(total_mask_loss/total_mask_element)**0.5}
        write(epoch_log_name,epoch_log)
        print(epoch_log)

    def save(self, epoch, file_path="output/bert_trained.model"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + ".ep%d.pth" % epoch
        torch.save(self.bert.cpu(), output_path)
        self.bert.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path
