import sys
import os
import argparse

sys.path.append(os.path.join(sys.path[0], '../'))

import torch
from compert.train import prepare_compert, evaluate_r2_pearson
import pandas as pd

parser = argparse.ArgumentParser(description='Postprocessing.')
parser.add_argument('--model_name',
                    type=str,
                    default='output/noctl_bert_pre/seed_0_best_model.pt')
parser.add_argument('--dataset_path',
                    type=str,
                    default='datasets/sciplex3_old_reproduced.h5ad')
parser.add_argument('--metr',
                    type=str,
                    default='distribution')
input_args = parser.parse_args()
metr = input_args.metr

spearman = False
model_name = input_args.model_name
out_dir = 'results/'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

state, args, history = torch.load(model_name, map_location=torch.device('cuda:0'))
args['dataset_path'] = input_args.dataset_path
if model_name == "output/split_mask40whole_lstm/seed_0_best_model.pt":
    args["encoder"] = "pretrain/whole.squ1.bert.ep200.pth"

if model_name == 'output/ori_sciplex3/model_seed=14_epoch=340.pt':
    args["split_key"] = 'split'

# load the dataset and model pre-trained weights
autoencoder, datasets = prepare_compert(args, state_dict=state)
autoencoder.eval()

if spearman:
    out_name = out_dir + metr+'spearman_' + model_name.replace('/', '_').replace(
        '.pt', '.csv')
else:
    out_name = out_dir +metr+ 'pearson_' + model_name.replace('/', '_').replace(
        '.pt', '.csv')
with torch.no_grad():
    df_ood = evaluate_r2_pearson(autoencoder,
                                 datasets['ood'],
                                 datasets['test_control'],
                                 full_report=True,
                                 metr=metr)
    df_ood['benchmark'] = autoencoder.model

    df_ood['metr'] = metr
    df_score = pd.concat([df_ood])

    df_score.to_csv(out_name, index=None)
