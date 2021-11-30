# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from model.ae_bert import PreAE
import os
import json
import argparse
import pdb
import tqdm
import sys

import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr
import torch.nn.functional as F

sys.path.append(os.path.join(sys.path[0], '../'))

from compert.data import load_dataset_splits, get_cell_encoding, get_drug_encoding
from compert.model import ComPert
from cvae import conditional_regressor
from cvae import torch_ode2cvae_cbnn_with_masking
from BERT import finetune
from sklearn.metrics import r2_score, balanced_accuracy_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from model.autoencoder import MLP

import time
from typing import *
from torch import Tensor


def pjson(s):
    """
    Prints a string in JSON format and flushes stdout
    """
    print(json.dumps(s), flush=True)


def evaluate_disentanglement(autoencoder, dataset, nonlinear=False):
    """
    Given a ComPert model, this function measures the correlation between
    its latent space and 1) a dataset's drug vectors 2) a datasets covariate
    vectors.
    """
    _, latent_basal = autoencoder.predict(dataset.genes,
                                          dataset.drugs,
                                          dataset.cell_types,
                                          return_latent_basal=True)

    latent_basal = latent_basal.detach().cpu().numpy()

    if nonlinear:
        clf = KNeighborsClassifier(n_neighbors=int(np.sqrt(len(latent_basal))))
    else:
        clf = LogisticRegression(solver="liblinear",
                                 multi_class="auto",
                                 max_iter=10000)

    pert_scores = cross_val_score(clf,
                                  StandardScaler().fit_transform(latent_basal),
                                  dataset.drugs_names,
                                  scoring=make_scorer(balanced_accuracy_score),
                                  cv=5,
                                  n_jobs=-1)

    if len(np.unique(dataset.cell_types_names)) > 1:
        cov_scores = cross_val_score(
            clf,
            StandardScaler().fit_transform(latent_basal),
            dataset.cell_types_names,
            scoring=make_scorer(balanced_accuracy_score),
            cv=5,
            n_jobs=-1)
        return np.mean(pert_scores), np.mean(cov_scores)
    else:
        return np.mean(pert_scores), 0


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
    """
    Measures different quality metrics about an ComPert `autoencoder`, when
    tasked to translate some `genes_control` into each of the drug/cell_type
    combinations described in `dataset`.
    Considered metrics are R2 score about means and variances for all genes, as
    well as R2 score about means and variances about differentially expressed
    (_de) genes.
    """
    autoencoder.eval()
    pearson_score, pearson_score_de = [], []
    pearson_score_de10 = []
    if autoencoder.model in ['AE']:
        genes_control, emb_drugs_ctl, emb_cts_ctl = [], [], []
        for gene, drug, cts in datasets_control:
            genes_control.append(gene)
            emb_drugs_ctl.append(drug)
            emb_cts_ctl.append(cts)
        genes_control = torch.stack(genes_control, dim=0)
        emb_drugs_ctl = torch.stack(emb_drugs_ctl, dim=0)
        emb_cts_ctl = torch.stack(emb_cts_ctl, dim=0)
        num, dim = genes_control.size(0), genes_control.size(1)

    elif autoencoder.model != 'codegen':
        genes_control = datasets_control.genes
        num, dim = genes_control.size(0), genes_control.size(1)
        if autoencoder.model == 'cvae':
            emb_drugs_ctl = datasets_control.drugs
            emb_cts_ctl = datasets_control.cell_types

    total_cells = len(dataset)
    category_list = []
    for pert_category in tqdm.tqdm(np.unique(dataset.pert_categories),
                              bar_format="{l_bar}{r_bar}"):
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
            if autoencoder.model in ['AE']:
                drug, cell = dataset.onehot(dataset.drugs_names[idx][0], dataset.cell_types_names[idx][0], dataset.dose_names[idx][0])
                emb_drugs = drug.view(1,-1).repeat(num,1).clone()
                emb_cts = cell.view(1, -1).repeat(num,1).clone()
                y_true = dataset.genes[idx]
            elif autoencoder.model != 'codegen':
                emb_drugs = dataset.drugs[idx][0].view(1,-1).repeat(num,1).clone()
                emb_cts = dataset.cell_types[idx][0].view(1, -1).repeat(num,1).clone()
                # estimate metrics only for reasonably-sized drug/cell-type combos
                y_true = dataset.genes[idx]

            else:
                cell_id, drug_id, dose = pert_category.split('_')
                emb_cts_onehot = torch.stack([get_cell_encoding(dataset, cell_id)])
                emb_drug_onehot = torch.stack([get_drug_encoding(dataset, drug_id)])
                emb_dosages_one = torch.Tensor([float(dose)])
                y_true = np.stack(dataset.df.iloc[idx, :]['expression'].values)

            # true means and variances
            yt_m = y_true.mean(axis=0)
            #yt_v = y_true.var(axis=0)

            if autoencoder.model == 'cpa':
                genes_predict = autoencoder.predict(genes_control, emb_drugs,
                                                    emb_cts).detach().cpu()

                mean_predict = genes_predict[:, :dim]
                #var_predict = genes_predict[:, dim:]

                # predicted means and variances
                yp_m = mean_predict.mean(0)
                #yp_v = var_predict.mean(0)

            elif autoencoder.model in ['BERT']:
                genes_control, emb_drugs, emb_cts = autoencoder.move_inputs_([genes_control, 
                    emb_drugs, emb_cts])
                mean_predict = autoencoder.forward(
                    genes_control, 
                    [emb_drugs, emb_cts]).detach().cpu()
                yp_m = mean_predict.mean(0)

            elif autoencoder.model in ['cvae','AE']:
                mean_predict = autoencoder.generate(
                    genes_control, [emb_drugs_ctl, emb_cts_ctl],
                    [emb_drugs, emb_cts]).detach().cpu()
                yp_m = mean_predict.mean(0)

            elif autoencoder.model == 'codegen':
                datasets_control.N = y_true.shape[0]//10 + 1

                loader_control = torch.utils.data.DataLoader(
                    datasets_control,
                    batch_size=autoencoder.hparams["batch_size"],
                    shuffle=True)
                mean_predict = []
                for genes_control, emb_drugs_ctl, dosages_ctl, emb_cts_ctl, mask_ctl in loader_control:
                    num, t = emb_drugs_ctl.shape[0], emb_drugs_ctl.shape[1]
                    emb_cts = torch.stack([emb_cts_onehot for _ in range(num)])
                    emb_drug_id = torch.stack([emb_drug_onehot for _ in range(num)])
                    emb_dosages = torch.stack([emb_dosages_one for _ in range(num)])
                    val_mask = torch.ones((num, t))
                    part_predict, val_err = autoencoder.mean_rec(genes_control, [emb_drugs_ctl, emb_cts_ctl],[emb_drug_id, emb_cts],emb_dosages,val_mask,dt=1,t_step=0.001,method='rk4')
                    mean_predict.append(part_predict.detach().cpu().reshape((num,genes_control.shape[-1])))
                mean_predict = np.concatenate(mean_predict)
                yp_m = mean_predict.mean(0)
            else:
                raise NotImplementedError

            if metr == 'mean':
                try:
                    mean_pearson = pearsonr(yt_m, yp_m)
                    mean_pearson_de = pearsonr(yt_m[de_idx], yp_m[de_idx])
                    pearson_score.append(mean_pearson[0])
                    pearson_score_de.append(mean_pearson_de[0])
                except:
                    print("Error when calculating mean_pearson",yt_m,yp_m)

                if dataset.de_genes_10:
                    mean_pearson_de10 = pearsonr(yt_m[de_idx10], yp_m[de_idx10])
                    pearson_score_de10.append(mean_pearson_de10[0])

            elif metr == 'distribution':
                mean_pearson = evaluate_distribution(y_true, mean_predict)
                mean_pearson_de = evaluate_distribution(
                    y_true[:, de_idx], mean_predict[:, de_idx])
                pearson_score.append(mean_pearson)
                pearson_score_de.append(mean_pearson_de)

                if dataset.de_genes_10:
                    mean_pearson_de10 = evaluate_distribution(
                        y_true[:, de_idx10], mean_predict[:, de_idx10])
                    pearson_score_de10.append(mean_pearson_de10)
    if full_report:
        print([
            np.nanmean(s) if len(s) else -1
            for s in [pearson_score, pearson_score_de, pearson_score_de10]
        ])
        if len(pearson_score_de10):
            return pd.DataFrame({
                'condition': category_list,
                'pearson': pearson_score,
                'pearson_DE': pearson_score_de,
                'pearson_DE10': pearson_score_de10
            })
        else:
            return pd.DataFrame({
                'condition': category_list,
                'pearson': pearson_score,
                'pearson_DE': pearson_score_de
            })

    """tmp_dir = f"result_{time.time()}.csv"
    print("DEBUG: save pearson result to",tmp_dir)    
    pd.DataFrame({
            'condition': category_list,
            'pearson': pearson_score,
            'pearson_DE': pearson_score_de,
            'pearson_DE10': pearson_score_de10
        }).to_csv(tmp_dir)"""

    return [
        np.nanmean(s) if len(s) else -1
        for s in [pearson_score, pearson_score_de, pearson_score_de10]
    ]


def evaluate(autoencoder, datasets):
    """
    Measure quality metrics using `evaluate()` on the training, test, and
    out-of-distributiion (ood) splits.
    """
    autoencoder.eval()
    with torch.no_grad():
        if autoencoder.model == 'cpa':
            stats_disent_pert, stats_disent_cov = evaluate_disentanglement(
                autoencoder, datasets["test"])

            evaluation_stats = {
                "training":
                None,
                #evaluate_r2_pearson(autoencoder, datasets["training_treated"],
                #                    datasets["training_control"]),
                "test":
                evaluate_r2_pearson(autoencoder, datasets["test_treated"],
                                         datasets["test_control"]),
                "ood":
                evaluate_r2_pearson(autoencoder, datasets["ood"],
                                    datasets["test_control"]),
                "perturbation disentanglement":
                stats_disent_pert,
                "optimal for perturbations":
                1 / datasets['test'].num_drugs,
                "covariate disentanglement":
                stats_disent_cov,
                "optimal for covariates":
                1 / datasets['test'].num_cell_types,
            }
        else:
            evaluation_stats = {
                "training":
                None,
                #evaluate_r2_pearson(autoencoder, datasets["training_treated"],
                #                    datasets["training_control"]),
                "test":
                None,
                "ood":
                evaluate_r2_pearson(autoencoder, datasets["ood"],
                                    datasets["test_control"]),
                #"optimal for perturbations":
                #1 / datasets['test'].num_drugs,
                #"optimal for covariates":
                #1 / datasets['test'].num_cell_types,
            }

    autoencoder.train()
    return evaluation_stats


def prepare_compert(args, state_dict=None):
    """
    Instantiates autoencoder and dataset to run an experiment.
    """
    if 'device' in args:
        device = args['device']
    else:
        device = 'cuda:0'
    if 'model' not in args:
        args['model'] = 'cpa'

    if args['doser_type'] in ['sigm', 'logsigm']:
        logdose = True
    elif args['doser_type'] in ['mlp', 'linear']:
        logdose = False
    else:
        raise ValueError("doser_type should be 'sigm', 'logsigm', 'mlp', 'linear'.")

    datasets = load_dataset_splits(args["dataset_path"],
                                   args["perturbation_key"], args["dose_key"],
                                   args["cell_type_key"], args["split_key"],model=args["model"], logdose=logdose)

    if args['model'] == 'cpa':
        autoencoder = ComPert(
            datasets["training"].num_genes,
            datasets["training"].num_drugs,
            datasets["training"].num_cell_types,
            device=device,
            seed=args["seed"],
            loss_ae=args["loss_ae"],
            doser_type=args["doser_type"],
            patience=args["patience"],
            hparams=args["hparams"],
            decoder_activation=args["decoder_activation"],
        )
    elif args['model'] == 'cvae':
        autoencoder = conditional_regressor.cVAE_Sequence(
            input_size=datasets["training"].num_genes,
            num_classes=2,
            class_sizes=[
                datasets["training"].num_drugs,
                datasets["training"].num_cell_types
            ],
            device=device,
            seed=args['seed'],
            hparams=args["hparams"]).to(device)
    elif args['model'] == 'codegen':
        net = conditional_regressor.cVAE_Sequence(
            input_size=datasets["training"].num_genes,
            num_classes=2,
            class_sizes=[
                datasets["training"].num_drugs,
                datasets["training"].num_cell_types
            ],
            device=device,
            seed=args['seed'],
            hparams=args["hparams"]).to(device)

        autoencoder = torch_ode2cvae_cbnn_with_masking.ODE2cVAERegression(
            cReg=net,
            q_ode=args["latent_dim_ode"],
            device=args["device"],
            use_full_second_order=args["use_full_second_order"],
            reg_freq=5,
            cbnn_num_hidden_layers=args["bnn_num_hidden_layers"],
            cbnn_hidden=args["bnn_hidden_dims"],
            seed=args['seed'],
            hparams=args["hparams"]).to(device)
        autoencoder.beta = args["beta"]
        autoencoder.gamma = args["gamma"]
  
    elif args['model'] == 'BERT':
        autoencoder = finetune.PreAE(
            input_size=datasets["training"].num_genes,
            num_classes=2,
            class_sizes=[
                datasets["training"].num_drugs,
                datasets["training"].num_cell_types
            ],
            device=device,
            seed=args['seed'],
            hparams=args["hparams"],
            encoder=args["encoder"]).to(device)

    elif args["model"] == "AE":
        autoencoder = torch.load(args["encoder"])
        autoencoder.model = "AE"
        num_genes = datasets["training"].num_genes

        #for param in autoencoder.parameters():
        #    param.requires_grad = False
        
        #autoencoder.gene_embeddings = torch.nn.Sequential(torch.nn.Linear(num_genes,autoencoder.hparams["gene_dim"]),torch.nn.Tanh())
        #autoencoder.gene_decoder = torch.nn.Linear(autoencoder.hparams["autoencoder_width"], num_genes)
        autoencoder.expr_encoder = torch.nn.Linear(num_genes,autoencoder.input_size)
        autoencoder.expr_decoder = torch.nn.Linear(autoencoder.input_size, num_genes)

        autoencoder.optimizer_autoencoder= torch.optim.Adam(
            autoencoder.parameters(),lr=1e-4,weight_decay=autoencoder.hparams["autoencoder_wd"])
        #for param_group in autoencoder.optimizer_autoencoder.param_groups:
        #    param_group["lr"] = 5e-4
        autoencoder = autoencoder.to(device)


    if state_dict is not None:
        autoencoder.load_state_dict(state_dict)
    print(autoencoder)
    return autoencoder, datasets


def train_compert(args, return_model=False):
    """
    Trains a ComPert autoencoder
    """

    autoencoder, datasets = prepare_compert(args)
    device = args['device']

    # batch_size is part of autoencoder.hparams
    datasets.update({
        "loader_tr":
        torch.utils.data.DataLoader(
            datasets["training"],
            batch_size=autoencoder.hparams["batch_size"],
            shuffle=True)
    })
    datasets.update({
        "loader_test":
        torch.utils.data.DataLoader(
            datasets["test"],
            batch_size=autoencoder.hparams["batch_size"],
            shuffle=True)
    })

    pjson({"training_args": args})
    pjson({"autoencoder_params": autoencoder.hparams})
    if not os.path.exists(args["save_dir"]):
        os.makedirs(args["save_dir"])

    start_time = time.time()
    val_mse = []
    best_model_loss = None
    best_model_epoch = None
    best_score = None

    for epoch in range(args["max_epochs"]):
        epoch_training_stats = defaultdict(float)
        autoencoder.train()
        torch.set_grad_enabled(True)
        print("training ...")
        for gene_drug_cell in tqdm.tqdm(datasets["loader_tr"],
                              bar_format="{l_bar}{r_bar}"):
            if args['model'] != 'codegen':
                minibatch_training_stats = autoencoder.update(*gene_drug_cell,train=True)
            else:
                # drug_id: [N, T, 132]
                # cell_id: [N, T, 9]
                # dosages: [N, T]
                # expr: [N, T, 978]
                # mask: [N, T]
                minibatch_training_stats = autoencoder.update(
                    *(gene_drug_cell + [len(datasets["loader_tr"])]))
            #if args['model'] == 'AE':
            #    minibatch_training_stats = {"loss": minibatch_training_stats}
            for key, val in minibatch_training_stats.items():
                epoch_training_stats[key] += val

        for key, val in epoch_training_stats.items():
            epoch_training_stats[key] = val / len(datasets["loader_tr"])
            if not (key in autoencoder.history.keys()):
                autoencoder.history[key] = []
            autoencoder.history[key].append(val)
        autoencoder.history['epoch'].append(epoch)

        ellapsed_minutes = (time.time() - start_time) / 60
        autoencoder.history['elapsed_time_min'] = ellapsed_minutes

        # decay learning rate if necessary
        # also check stopping condition: patience ran out OR
        # time ran out OR max epochs achieved
        stop = ellapsed_minutes > args["max_minutes"] or \
            (epoch == args["max_epochs"] - 1)

        if (epoch % args["checkpoint_freq"]) == 0 or stop:
            print("evaluating ...")
            evaluation_stats = evaluate(autoencoder, datasets)
            for key, val in evaluation_stats.items():
                if not (key in autoencoder.history.keys()):
                    autoencoder.history[key] = []
                autoencoder.history[key].append(val)
            autoencoder.history['stats_epoch'].append(epoch)
            pjson({"epoch": epoch,"training_stats": epoch_training_stats,"evaluation_stats": evaluation_stats,"ellapsed_minutes": ellapsed_minutes})

            torch.save(
                (autoencoder.state_dict(), args, autoencoder.history),
                os.path.join(
                    args["save_dir"],
                    "model_seed={}_epoch={}.pt".format(args["seed"], epoch)))

            pjson({
                "model_saved":
                "model_seed={}_epoch={}.pt\n".format(args["seed"], epoch)
            })
            if args['model'] == 'cpa':
                this_score = evaluation_stats["test"][0]+\
                    evaluation_stats["test"][1]+\
                    evaluation_stats["test"][2]
                this_score -=  abs(evaluation_stats["perturbation disentanglement"] -\
                        evaluation_stats["optimal for perturbations"])/2 +\
                        abs(evaluation_stats["covariate disentanglement"] -\
                        evaluation_stats["optimal for covariates"])/2
                if best_score is None or this_score > best_score:
                    best_score = this_score
                    best_model_epoch = epoch
                    torch.save(
                        (autoencoder.state_dict(), args, autoencoder.history),
                        os.path.join(
                            args["save_dir"],
                            "seed_{}_best_model.pt".format(args["seed"])))
                    print('\tupdated to new best model at epoch', epoch)

                stop = stop or autoencoder.early_stopping(
                    np.mean(evaluation_stats["test"]))

            elif args['model'] in ['cvae','AE']:
                for param in autoencoder.expr_encoder.parameters():
                    print("expr_encoder",param)
                with torch.set_grad_enabled(False):
                    for genes, drugs, cell_types in datasets['loader_test']:
                        outputs = autoencoder(genes, [drugs, cell_types])
                        val_loss = autoencoder.loss_fn(*outputs, beta=1.0)
                        val_mse.append(val_loss['MSE'].item())
                        break

                if best_model_loss is None or val_mse[-1] < best_model_loss:
                    print('\tupdated to new best model at epoch', epoch)
                    best_model_loss = val_mse[-1]
                    pjson({"best_model_loss": best_model_loss})
                    best_model_epoch = epoch
                    torch.save(
                        (autoencoder.state_dict(), args, autoencoder.history),
                        os.path.join(
                            args["save_dir"],
                            "seed_{}_best_model.pt".format(args["seed"])))

                stop = stop or autoencoder.check_for_early_break_condition(
                    val_mse)

            elif args['model'] == 'codegen':
                with torch.set_grad_enabled(False):
                    for val_expr, val_drug, val_dosages, val_cell, val_mask in datasets[
                            'loader_test']:

                        Xrec_mu, val_err = autoencoder.mean_rec(
                            val_expr, [val_drug, val_cell],
                            [val_drug, val_cell],
                            val_dosages,
                            val_mask,
                            dt=1,
                            t_step=0.001,
                            method='rk4')
                        val_mse.append(val_err.item())
                        break

                if best_model_loss is None or val_mse[-1] < best_model_loss:
                    print('\tupdated to new best model at epoch', epoch)
                    best_model_loss = val_mse[-1]
                    best_model_epoch = epoch
                    pjson({"best_model_loss": best_model_loss, 'best_model_epoch': epoch})
                    torch.save(
                        (autoencoder.state_dict(), args, autoencoder.history),
                        os.path.join(
                            args["save_dir"],
                            "seed_{}_best_model.pt".format(args["seed"])))

                stop = stop or autoencoder.check_for_early_break_condition(
                    val_mse)

            elif args['model'] in ['BERT']:

                with torch.set_grad_enabled(False):
                    for genes, drugs, cell_types in datasets['loader_test']:
                        genes, drugs, cell_types = autoencoder.move_inputs_([genes, drugs, cell_types])
                        outputs = autoencoder(genes, [drugs, cell_types])
                        val_loss = F.mse_loss(outputs, genes, reduction='mean')
                        val_mse.append(val_loss.item())
                        break

                if best_model_loss is None or val_mse[-1] < best_model_loss:
                    print('\tupdated to new best model at epoch', epoch)
                    best_model_loss = val_mse[-1]
                    pjson({"best_model_loss": best_model_loss})
                    best_model_epoch = epoch
                    torch.save(
                        (autoencoder.state_dict(), args, autoencoder.history),
                        os.path.join(
                            args["save_dir"],
                            "seed_{}_best_model.pt".format(args["seed"])))

                stop = stop or autoencoder.check_for_early_break_condition(
                    val_mse)

            if stop:
                pjson({"early_stop": epoch})
                break
        pjson({"lr":autoencoder.optimizer_autoencoder.state_dict()['param_groups'][0]['lr']})
        #autoencoder.scheduler_autoencoder.step()
    pjson({'best_model': best_model_epoch})
    if return_model:
        return autoencoder, datasets


def parse_arguments():
    """
    Read arguments if this script is called from a terminal.
    """

    parser = argparse.ArgumentParser(description='Drug combinations.')
    # dataset arguments
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--perturbation_key', type=str, default="condition")
    parser.add_argument('--dose_key', type=str, default="dose_val")
    parser.add_argument('--cell_type_key', type=str, default="cell_type")
    parser.add_argument('--split_key', type=str, default="split")
    parser.add_argument('--loss_ae', type=str, default='gauss')
    parser.add_argument('--doser_type', type=str, default='sigm')
    parser.add_argument('--decoder_activation', type=str, default='linear')

    # ComPert arguments (see set_hparams_() in compert.model.ComPert)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hparams', type=str, default="")

    # training arguments
    parser.add_argument('--max_epochs', type=int, default=2000)
    parser.add_argument('--max_minutes', type=int, default=1440)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--checkpoint_freq', type=int, default=1)

    # output folder
    parser.add_argument('--save_dir', type=str, required=True)
    # number of trials when executing compert.sweep
    parser.add_argument('--sweep_seeds', type=int, default=200)

    # Model
    parser.add_argument('--model', type=str, default='cpa', 
        choices=["cpa","cvae","codegen","BERT","AE"])
    parser.add_argument('--encoder', type=str, default=None,
                    help='Encoder for BERT and AE')

    # CodeGen
    parser.add_argument('--latent-dim-ode',
                        type=int,
                        default=4,
                        help='Dimension of latent space for ODE operation')
    parser.add_argument('--use-full-second-order',
                        action='store_true',
                        help='Use the full (slower) second order ODE')
    parser.add_argument('--bnn-num-hidden-layers',
                        type=int,
                        default=3,
                        help='Number of hidden layers for BNN or cBNN')
    parser.add_argument('--bnn-hidden-dims',
                        type=int,
                        default=50,
                        help='Size of hidden layers for BNN or cBNN')
    parser.add_argument('--beta',
                        type=float,
                        default=1.0,
                        help='Regularization weight for f_W')
    parser.add_argument(
        '--gamma',
        type=float,
        default=1.0,
        help='Regularization weight for secondary sequence likelihood')

    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help="Device to use")

    return dict(vars(parser.parse_args()))


if __name__ == "__main__":
    args = parse_arguments()
    train_compert(args)
