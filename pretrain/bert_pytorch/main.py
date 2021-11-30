import argparse
import os
import pdb
import random
import torch
import sys

from torch.utils.data import DataLoader
import numpy as np
from torch.optim import Adam

from model import BERT
from trainer import BERTTrainer
from dataset import BERTDataset


def train():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--train_dataset", default="../../dataset/trt_cp_train.gctx", type=str, help="train dataset for train bert")
    #parser.add_argument("-t", "--test_dataset", type=str, default=None, help="test set for evaluate train set")
    parser.add_argument("-t", "--test_dataset", type=str, default="../../dataset/trt_cp_test.gctx", help="test set for evaluate train set")
    parser.add_argument("-cv", "--cell_vocab_path", default="../cell_vocab.txt", type=str, help="built vocab model path with bert-vocab")
    parser.add_argument("-dv", "--drug_vocab_path", default="../drug_vocab.txt", type=str, help="built vocab model path with bert-vocab")
    parser.add_argument("-o", "--output_path", default="output/oneword_1024_2_2/", type=str, help="ex)output/bert.model")

    parser.add_argument("-hs", "--hidden", type=int, default=1024, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=2, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=2, help="number of attention heads")
    #parser.add_argument("-s", "--seq_len", type=int, default=978, help="maximum sequence len")

    parser.add_argument("-b", "--batch_size", type=int, default=2048, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=0, help="dataloader worker size")

    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--log_freq", type=int, default=10, help="printing loss every n iter: setting n")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")

    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

    parser.add_argument("--resume", type=str, default="-1", help="Resume from which epoch")
    parser.add_argument("--model_path", type=str, default="", help="For debug, resume from this model")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()
    if not args.output_path.endswith('/'):
        args.output_path += '/'
    args.log_name = args.output_path
    print(args)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    #torch.backends.cudnn.deterministic = True
    random.seed(args.seed)

    output_path=args.output_path.rsplit('/',1)[0]
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    """print("Loading Cell Vocab", args.cell_vocab_path)
    with open(args.cell_vocab_path) as f:
        cell_vocab = f.read().split('\n')
    print("Vocab Size: ", len(cell_vocab))
    
    print("Loading Drug Vocab", args.drug_vocab_path)
    with open(args.drug_vocab_path) as f:
        drug_vocab = f.read().split('\n')
    print("Vocab Size: ", len(drug_vocab))"""

    #kmeans_label = np.load(args.kmeans_labels_path)
    print("Loading Train Dataset", args.train_dataset)
    train_dataset = BERTDataset(args.train_dataset)
    args.seq_len = train_dataset.seq_len

    print("Loading Test Dataset", args.test_dataset)
    test_dataset = BERTDataset(args.test_dataset) \
        if args.test_dataset is not None else None

    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers) \
        if test_dataset is not None else None

    max_step = args.epochs*len(train_data_loader)

    if args.model_path != "":
        print("DEBUG: Load model from", args.model_path)
        bert = torch.load(args.model_path.replace("lm","bert"))
        model = torch.load(args.model_path)
        trainer = BERTTrainer(bert, args.seq_len,train_dataloader=train_data_loader, test_dataloader=test_data_loader,
            lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
            with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq, log_name=args.log_name,resume=model,max_step=max_step)

    elif os.path.exists(args.output_path+"bert.ep"+args.resume+".pth"):
        bert = torch.load(args.output_path+"bert.ep"+args.resume+".pth")
        optim=None
        if os.path.exists(args.output_path+"lm.ep"+args.resume+".pth"):
            print("Load model from",args.output_path+"lm.ep"+args.resume+".pth")
            model = torch.load(args.output_path+"lm.ep"+args.resume+".pth")   
        if os.path.exists(args.output_path+"optim.ep"+args.resume+".pth"):
            print("Load model from",args.output_path+"optim.ep"+args.resume+".pth")
            optim = Adam(model.parameters())
            optim.load_state_dict(torch.load(args.output_path+"optim.ep"+args.resume+".pth"))   
    
        trainer = BERTTrainer(bert, args.seq_len,train_dataloader=train_data_loader, test_dataloader=test_data_loader,
            lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
            with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq, log_name=args.log_name,resume=model,optim=optim,
            resume_epoch=int(args.resume),max_step=max_step)

    else:
        args.resume = "-1"
        print("Building BERT model")
        bert = BERT(hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads, seq_len=args.seq_len)

        print("Creating BERT Trainer")
        trainer = BERTTrainer(bert, args.seq_len,train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                            lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                            with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq, log_name=args.log_name,max_step=max_step)

    print("Training Start")
    for epoch in range(int(args.resume)+1, args.epochs):
        trainer.train(epoch)
        trainer.save(epoch, args.output_path)

        if test_data_loader is not None:
            trainer.test(epoch)

if __name__ == "__main__":
    train()

