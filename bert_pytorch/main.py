import argparse
import os
import pdb
import random
import torch

from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from torch.optim import Adam

from model import BERT, BERTLM
from trainer import BERTTrainer
from dataset import BERTDataset


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
    test_dataset = BERTDataset(args.test_dataset, cell_vocab,drug_vocab, gene_thre) \
        if args.test_dataset is not None else None

    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers) \
        if test_dataset is not None else None

    if os.path.exists(args.output_path+"bert.ep"+args.resume+".pth"):
        bert = torch.load(args.output_path+"bert.ep"+args.resume+".pth")
        optim=None
        if os.path.exists(args.output_path+"lm.ep"+args.resume+".pth"):
            print("Load model from",args.output_path+"lm.ep"+args.resume+".pth")
            model = torch.load(args.output_path+"lm.ep"+args.resume+".pth")   
        if os.path.exists(args.output_path+"optim.ep"+args.resume+".pth"):
            print("Load model from",args.output_path+"optim.ep"+args.resume+".pth")
            optim = Adam(model.parameters())
            optim.load_state_dict(torch.load(args.output_path+"optim.ep"+args.resume+".pth"))   
    
        trainer = BERTTrainer(bert, args.gen_len,len(cell_vocab),len(drug_vocab), args.seq_len,train_dataloader=train_data_loader, test_dataloader=test_data_loader,
            lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
            with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq, log_name=args.log_name,resume=model,optim=optim)

    else:
        args.resume = "-1"
        print("Building BERT model")
        bert = BERT(args.gen_len,hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads)

        print("Creating BERT Trainer")
        trainer = BERTTrainer(bert, args.gen_len,len(cell_vocab),len(drug_vocab), args.seq_len,train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                            lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                            with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq, log_name=args.log_name)

    print("Training Start")
    for epoch in range(int(args.resume)+1, args.epochs):
        trainer.train(epoch)
        trainer.save(epoch, args.output_path)

        if test_data_loader is not None:
            trainer.test(epoch)

if __name__ == "__main__":
    train()

