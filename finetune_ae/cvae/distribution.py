import os
import json
import argparse
import pdb
import sys

import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr

sys.path.append(os.path.join(sys.path[0], '../'))
import torch
from compert.train import prepare_compert
import pandas as pd
from compert.data import load_dataset_splits
from compert.model import ComPert
from cvae import conditional_regressor
from cvae import torch_ode2cvae_cbnn_with_masking

from sklearn.metrics import r2_score, balanced_accuracy_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def evaluate_distribution(y_true, mean_predict, sample_size=30):
    #print('y_true',y_true.shape,'mean_predict',mean_predict.shape)
    sample_size1 = min(sample_size, y_true.shape[0])
    sample_size2 = min(sample_size, mean_predict.shape[0])
    y_sample = np.random.choice(y_true.shape[0], sample_size1, False)
    corr = np.zeros(y_sample.shape[0])
    predict_sample = np.random.choice(mean_predict.shape[0], sample_size2,
                                      False)
    mean_predict = mean_predict[predict_sample, :]
    for i, cell in enumerate(y_true[y_sample, :]):
        true_corr = np.zeros(predict_sample.shape[0])
        for j, predict_cell in enumerate(mean_predict):
            true_corr[j] = pearsonr(cell, predict_cell)[0]
        corr[i] = np.nanmax(true_corr)
    corr_mean = np.nanmean(corr)
    return corr_mean


def evaluate_r2_pearson(autoencoder,
                        dataset,
                        datasets_control,
                        full_report=False,
                        metr='mean'):
   
    pearson_score, pearson_score_de = [], []
    pearson_score_de10 = []
    genes_control = datasets_control.genes
    num, dim = genes_control.size(0), genes_control.size(1)
    emb_drugs_ctl = datasets_control.drugs
    emb_cts_ctl = datasets_control.cell_types

    total_cells = len(dataset)
    category_list = []
    for pert_category in np.unique(dataset.pert_categories):
        # pert_category category contains: 'celltype_perturbation_dose' info
        de_idx = np.where(
            dataset.var_names.isin(np.array(
                dataset.de_genes[pert_category])))[0]
        if dataset.de_genes_10:
            de_idx10 = np.where(
                dataset.var_names.isin(np.array(
                    dataset.de_genes_10[pert_category])))[0]
        idx = np.where(dataset.pert_categories == pert_category)[0]
        if len(idx) > 0:
            category_list.append(pert_category)
            emb_drugs = dataset.drugs[idx][0].view(1,
                                                   -1).repeat(num,
                                                              1).clone()
            emb_cts = dataset.cell_types[idx][0].view(1, -1).repeat(num,
                                                                1).clone()
            # estimate metrics only for reasonably-sized drug/cell-type combos
            y_true = dataset.genes[idx]
                
            # true means and variances
            yt_m = y_true.mean(axis=0)
            #yt_v = y_true.var(axis=0)


            mean_predict = autoencoder.generate(
                genes_control, [emb_drugs_ctl, emb_cts_ctl],
                [emb_drugs, emb_cts]).detach().cpu()
            #yp_m = mean_predict.mean(0)
            
            '''
            mean_pearson = evaluate_distribution(y_true, mean_predict)
            mean_pearson_de = evaluate_distribution(
                y_true[:, de_idx], mean_predict[:, de_idx])

            if dataset.de_genes_10:
                mean_pearson_de10 = evaluate_distribution(
                    y_true[:, de_idx10], mean_predict[:, de_idx10])
                pearson_score_de10.append(mean_pearson_de10)
            '''
            print(y_true.shape)
            print(mean_predict.shape)


model_name = '/home/wsk/neuralode/CPA/zeroshot/zeroshot1/seed_0_best_model.pt'

state, args, history = torch.load(model_name, map_location=torch.device('cuda:0'))
#args['dataset_path'] = '../datasets/GSM_new.h5ad'
args['dataset_path'] = '../datasets/sciplex3_0727.h5ad'

# load the dataset and model pre-trained weights
autoencoder, datasets = prepare_compert(args, state_dict=state)
autoencoder.eval()
with torch.no_grad():
  #genes_control = datasets['training_control'].genes
  #df_train = evaluate_r2(autoencoder, datasets['training_treated'], genes_control)
  #df_train['benchmark'] = autoencoder.model
  
  genes_control = datasets['test_control'].de_genes_10
  #print(type(genes_control))
  '''
  df_continuous = evaluate_r2_pearson(autoencoder, datasets['continuous'], datasets['test_control'],full_report=True,metr="distribution")
  df_continuous['benchmark'] = autoencoder.model
  df_combination = evaluate_r2_pearson(autoencoder, datasets['combination'], datasets['test_control'],full_report=True,metr="distribution")
  df_combination['benchmark'] = autoencoder.model
  df_contcomb = evaluate_r2_pearson(autoencoder, datasets['contcomb'], datasets['test_control'],full_report=True,metr="distribution")
  df_contcomb['benchmark'] = autoencoder.model
  '''
  df_ood = evaluate_r2_pearson(autoencoder, datasets['ood'], datasets['test_control'],full_report=True,metr="distribution")
  #genes_control = datasets['test_control'].genes
  #df_test = evaluate_r2_pearson(autoencoder, datasets['test_treated'], genes_control,full_report=True)
  #df_test['benchmark'] = autoencoder.model
  
  #df_continuous['split'] = 'continous'
  #df_combination['split'] = 'combination'
  #df_contcomb['split'] = 'contcomb'
