import argparse
import os
import pdb
import random
import tqdm
import torch

from dataset.impute import ImputeDataset

from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from torch.optim import Adam
from torch import nn
from trainer.optim_schedule import ScheduledOptim
from model.imputation import Imputation

from model import BERT
import json




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
            self.model = Imputation(bert,gene_thre,seq_len)
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
            data = {key: value.to(self.device) for key, value in data.items()}

            # 1. forward the next_sentence_prediction and masked_lm model
            mask_lm_output = self.model.forward(data["bert_input"])

            # 2-2. MSELoss of predicting masked token word
            mask_loss = self.nll(mask_lm_output.transpose(1,2), data["bert_label"])

            # 2-3. Adding class_loss and mask_loss : 3.4 Pre-training Procedure
            loss = cell_loss + drug_loss + dose_loss + mask_loss
            #print('cell_loss',cell_loss.item(),'drug_loss',drug_loss.item(),'dose_loss',dose_loss.item(),'mask_loss',mask_loss.item())

            # 3. backward and optimization only in train
            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

            avg_loss += loss.item()

            for k,(output, class_type) in enumerate(zip([cell_output, drug_output], ["cell", "drug"])):
                correct = output.argmax(dim=-1).eq(data[class_type]).sum().item()
                total_correct[k] += correct
                total_element[k] += data[class_type].nelement()
            valid_idx = [data["bert_label"]!=0]
            correct = mask_lm_output[valid_idx].argmax(dim=-1).eq(data["bert_label"][valid_idx]).sum().item()
            total_correct[2] += correct
            total_element[2] += data["bert_label"][valid_idx].nelement()
            
            total_dose_loss += dose_loss.item()*data["dose"].nelement()
            total_dose_element += data["dose"].nelement()

            if i % self.log_freq == 0:
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
        #"total_mask_rmse":(total_mask_loss/total_mask_element)**0.5
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
        lm_output_path = file_path + "lm.ep%d.pth" % epoch
        torch.save(self.optim.state_dict(), lm_output_path.replace("lm","optim"))
        torch.save(self.model.cpu(), lm_output_path)
        self.model.to(self.device)
        print("EP:%d Model Saved on:" % epoch, bert_output_path, lm_output_path)
        return bert_output_path

def train():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--train_dataset", default="/rd1/user/tanyh/perturbation/dataset/trt_cp_landmarkonly_train.gctx", type=str, help="train dataset for train bert")
    parser.add_argument("-t", "--test_dataset", type=str, default="/rd1/user/tanyh/perturbation/dataset/trt_cp_landmarkonly_test.gctx", help="test set for evaluate train set")
    parser.add_argument("-cv", "--cell_vocab_path", default="/rd1/user/tanyh/perturbation/pretrain_BERT/cell_vocab.txt", type=str, help="built vocab model path with bert-vocab")
    parser.add_argument("-dv", "--drug_vocab_path", default="/rd1/user/tanyh/perturbation/pretrain_BERT/drug_vocab.txt", type=str, help="built vocab model path with bert-vocab")
    parser.add_argument("-gv", "--gene_thre_path", default="/rd1/user/tanyh/perturbation/pretrain_BERT/dist_info.csv", type=str, help="built vocab model path with bert-vocab")
    #parser.add_argument("-kl", "--kmeans_labels_path", default="/rd1/user/tanyh/perturbation/pretrain_BERT/kmeans_label.npy", type=str, help="kmeans_label.npy")
    parser.add_argument("-o", "--output_path", default="/rd1/user/tanyh/perturbation/pretrain_BERT/output/0821_128_2_2/", type=str, help="ex)output/")

    parser.add_argument("-hs", "--hidden", type=int, default=768, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=12, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=12, help="number of attention heads")

    parser.add_argument("-b", "--batch_size", type=int, default=64, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=300, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=5, help="dataloader worker size")

    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--log_freq", type=int, default=10, help="printing loss every n iter: setting n")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")

    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

    parser.add_argument("--resume", type=str, default="-1", help="Resume from which epoch")

    args = parser.parse_args()
    if not args.output_path.endswith('/'):
        args.output_path += '/'
    args.log_name = args.output_path
    print(args)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    seed=0
    torch.manual_seed(seed)
    np.random.seed(seed)
    #torch.backends.cudnn.deterministic = True
    random.seed(seed)


    print("Loading Cell Vocab", args.cell_vocab_path)
    with open(args.cell_vocab_path) as f:
        cell_vocab = f.read().split('\n')
    print("Vocab Cell Size: ", len(cell_vocab))
    
    print("Loading Drug Vocab", args.drug_vocab_path)
    with open(args.drug_vocab_path) as f:
        drug_vocab = f.read().split('\n')
    print("Vocab Drug Size: ", len(drug_vocab))

    print("Loading Gene Thre", args.gene_thre_path)
    gene_thre = pd.read_csv(args.gene_thre_path, index_col=0)

    print("Loading Train Dataset", args.train_dataset)
    train_dataset = BERTDataset(args.train_dataset, cell_vocab,drug_vocab, gene_thre)
    args.seq_len = train_dataset.seq_len
    args.gen_len = train_dataset.gen_len

    print("Loading Test Dataset", args.test_dataset)
    test_dataset = BERTDataset(args.test_dataset, cell_vocab,drug_vocab, gene_thre)

    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    if os.path.exists(args.output_path+"bert.ep"+args.resume+".pth"):
        bert = torch.load(args.output_path+"bert.ep"+args.resume+".pth")
         


if __name__ == "__main__":
    train()

