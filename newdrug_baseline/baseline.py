#%%
from data import Dataset, load_dataset_splits
import numpy as np
from scipy.stats import pearsonr
import pandas as pd
import torch
from scipy.stats import entropy
import time
import pdb

def evaluate_KL(y_true,mean_predict):
    scores = []
    for i in range(y_true.shape[1]):
        true_dist = [0.5]*21
        true_vec = y_true[:,i]
        true_vec = (torch.ceil(true_vec * 10))/10.0
        num1 = len(true_vec)
        for k in range(num1):
            pos = int(true_vec[k]*10)
            if pos > 20:
                true_dist[20] += 1
            else:
                true_dist[pos] += 1
        pred_dist = [0.5]*21
        pred_vec = mean_predict[:,i]
        pred_vec = (torch.ceil(pred_vec * 10))/10.0
        num2 = len(pred_vec)
        for t in range(num2):
            pos = int(pred_vec[t]*10)
            if pos > 20:
                pred_dist[20] += 1
            else:
                pred_dist[pos] += 1
        #pred_dist = pred_dist/sum(pred_dist)
        #print(true_dist,pred_dist)
        score = entropy(true_dist,pred_dist)
        #print(score)
        scores.append(score)
        #print(scores) 

    mean_score = np.nanmean(scores)
    return mean_score

def evaluate_distribution(y_true, mean_predict, sample_size=30):
    #print('y_true',y_true.shape,'mean_predict',mean_predict.shape)
    sample_size = min(sample_size, y_true.shape[0], mean_predict.shape[0])
    y_sample = np.random.choice(y_true.shape[0], sample_size, False)
    y_true = y_true[y_sample,:]
    corr = np.zeros(y_sample.shape[0])
    predict_sample = np.random.choice(mean_predict.shape[0], sample_size,
                                      False)
    mean_predict = mean_predict[predict_sample, :]
    for i, cell in enumerate(y_true):
        true_corr = np.zeros(predict_sample.shape[0])
        for j, predict_cell in enumerate(mean_predict):
            true_corr[j] = pearsonr(cell, predict_cell)[0]
        corr[i] = np.nanmax(true_corr)
    corr_mean = np.nanmean(corr)
    return corr_mean

def evaluate_mean(y_true,mean_predict):
    yt_m = y_true.mean(axis=0)
    yp_m = mean_predict.mean(0)
    mean_pearson = pearsonr(yt_m, yp_m)
    return mean_pearson[0]

if __name__ == "__main__":
    start = time.time()
    np.random.seed(1)

    # PARAM
    leaveoneout = False
    if leaveoneout:
        train = Dataset('/rd1/user/tanyh/perturbation/CPA/datasets/sciplex3_1016.h5ad')
        test = train

    else:
        split = load_dataset_splits('/rd1/user/tanyh/perturbation/CPA/datasets/sciplex3_1016.h5ad')
        train = split['train']
        test = split['test']
        train_drug = list(set(train.drugs_names))

    cos_weight=pd.read_csv('A549_0.1_cosine.csv',header=0,index_col=0)
    test_drugs_names = set(test.drugs_names) & set(cos_weight.index)

    # PARAM
    select_num = 5
    evaluate = 'distribution'

    print('select_num', select_num)
    print('leaveoneout', leaveoneout)
    print('evaluate', evaluate)

    from collections import defaultdict

    train_drug_dict = defaultdict(set)
    for i, drug in enumerate(train.drugs_names):
        train_drug_dict[drug].add(i)

    train_cell_dict = defaultdict(set)
    for i, cell in enumerate(train.cell_types_names):
        train_cell_dict[cell].add(i)

    train_dosage_dict = defaultdict(set)
    for i, dosage in enumerate(train.dose_names):
        train_dosage_dict[dosage].add(i)

    test_drug_dict = defaultdict(set)
    for i, drug in enumerate(test.drugs_names):
        test_drug_dict[drug].add(i)

    test_cell_dict = defaultdict(set)
    for i, cell in enumerate(test.cell_types_names):
        test_cell_dict[cell].add(i)

    test_dosage_dict = defaultdict(set)
    for i, dosage in enumerate(test.dose_names):
        test_dosage_dict[dosage].add(i)

    #%%
    drug_list = []
    cell_list = []
    dose_list = []
    evaluate_list = []
    de = []
    de10 = []
    all_drug = np.unique(train.drugs_names)

    for cell_type in np.unique(test.cell_types_names):
        for dose in np.unique(test.dose_names):
            
            for drug_name in test_drugs_names:
                if leaveoneout:
                    train_drug = list(set(all_drug[all_drug!=drug_name]) & set(cos_weight.index))

                true_idx = test_cell_dict[cell_type] & test_drug_dict[drug_name] & test_dosage_dict[dose]
                # True
                pert_category = '_'.join([cell_type, test.cmap2condition[drug], str(dose)])

                de_idx = np.where(test.var_names.isin(np.array(test.de_genes[pert_category])))[0]
                y_true_de = test.genes[list(true_idx)][:,de_idx]
                de_idx10 = np.where(test.var_names.isin(np.array(test.de_genes_10[pert_category])))[0]
                y_true_de10 = test.genes[list(true_idx)][:,de_idx10]

                y_true = test.genes[list(true_idx),:]
                y_true_len = y_true.shape[0]
                if y_true_len > 0:
                    random_drug = np.random.choice(train_drug, select_num)
                    predict_drug_idx = set()
                    for drug_idx in random_drug:
                        if drug_idx == 'DMSO':
                            predict_drug_idx = predict_drug_idx | (train_drug_dict[drug_idx] & train_dosage_dict[1.0])
                        else:
                            predict_drug_idx = predict_drug_idx | (train_drug_dict[drug_idx] & train_dosage_dict[dosage])
                    predict_idx = list(train_cell_dict[cell_type] & predict_drug_idx)
                    random_idx = np.random.choice(predict_idx, y_true_len)
                    # Predict
                    mean_predict = train.genes[random_idx,:]
                    mean_predict_de = mean_predict[:, de_idx]
                    mean_predict_de10 = mean_predict[:, de_idx10]
                    drug_list.append(drug_name)
                    cell_list.append(cell_type)
                    dose_list.append(dose)
                    
                    if evaluate=='distribution':
                        metr = evaluate_distribution
                    elif evaluate == 'KL':
                        metr = evaluate_KL
                    elif evaluate == 'mean':
                        metr = evaluate_mean
                    else:
                        raise NotImplementedError

                    evaluate_list.append(metr(y_true,mean_predict))
                    de.append(metr(y_true_de,mean_predict_de))
                    de10.append(metr(y_true_de10,mean_predict_de10))

    result = pd.DataFrame({'drug':drug_list,'cell_type':cell_list,'dose':dose_list,'whole':evaluate_list,
        'de':de, 'de10':de10})
    result.to_csv(f'baseline_{evaluate}_{select_num}_{leaveoneout}.csv')
    print('mean',np.nanmean(result['whole']), np.nanmean(result['de']),np.nanmean(result['de10']))
    print('time',time.time()-start)
