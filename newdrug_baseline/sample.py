  
#%%
import random
import numpy as np
import pandas as pd
from data import load_dataset_splits
from scipy.stats import pearsonr
from collections import defaultdict
import torch
from scipy.stats import entropy
from baseline import evaluate_distribution, evaluate_mean

random.seed(0)
np.random.seed(0)

split = load_dataset_splits('/rd1/user/tanyh/perturbation/CPA/datasets/sciplex3_1016.h5ad')
train = split['train']
test = split['test']
#metr = evaluate_distribution
metr = evaluate_mean
print(metr)
useDE = True
#%%
#cos_weight=pd.read_csv('/home/wsk/neuralode/CPA/zeroshot0805/lincs_sciplex_cosine.csv',header=0,index_col=0)
train_drug = np.unique(train.drugs_names)
test_drug = np.unique(test.drugs_names)
#cos_weight = cos_weight.loc[cos_weight.index.isin(test_drug), cos_weight.index.isin(train_drug)]
#%%
train_drug_dict = defaultdict(set)
for i, drug in enumerate(train.drugs_names):
    train_drug_dict[drug].add(i)
#print(train_drug_dict['LY-2784544'])

train_cell_types_dict = defaultdict(set)
for i, cell_type in enumerate(train.cell_types_names):
    train_cell_types_dict[cell_type].add(i)
#print(train_cell_types_dict)

train_dosage_dict = defaultdict(set)
for i, dosage in enumerate(train.dose_names):
    train_dosage_dict[dosage].add(i)
#print(train_dosage_dict)

test_drug_dict = defaultdict(set)
for i, drug in enumerate(test.drugs_names):
    test_drug_dict[drug].add(i)
#print(test_drug_dict['LY-2784544'])

test_cell_types_dict = defaultdict(set)
for i, cell_type in enumerate(test.cell_types_names):
    test_cell_types_dict[cell_type].add(i)
#print(test_cell_types_dict)

test_dosage_dict = defaultdict(set)
for i, dosage in enumerate(test.dose_names):
    test_dosage_dict[dosage].add(i)
#print(test_dosage_dict)



#%%
drug_names = np.unique(test.drugs_names) #得到要预测的药物名称集

dosage_list = np.unique(test.dose_names)
M = len(dosage_list)
#W = cos_weight.values #n*n matrix，距离矩阵
#N = len(W)
#index = cos_weight.index #series,索引是1-N，value是每行/列对应的药物名称


#%%

drug_list = []
cell_list = []
dose_list = []
evaluate_list = []
de = []
de10 = []

for cell_type in np.unique(test.cell_types_names):
    for dose in np.unique(test.dose_names):
        cos_weight=pd.read_csv(f'/rd1/user/tanyh/perturbation/newdrug_baseline/{cell_type}_{dose}_cosine.csv',header=0,index_col=0)
        cos_weight = cos_weight.loc[cos_weight.index.isin(test_drug), cos_weight.index.isin(train_drug)]
        for drug in np.unique(test.drugs_names):
            #drug_dosage_dict = {}
            if drug in cos_weight.index:
                drug_dist = cos_weight.loc[drug,:]
                drug_index = drug_dist.argsort()[:5]#drug name
                coeff = 1/drug_dist[drug_index]
                coeff = coeff/sum(coeff)
                ref_drugs_names = coeff.index
        
                idx = test_cell_types_dict[cell_type] & test_dosage_dict[dose] & test_drug_dict[drug]
                #print(idx)
                pert_category = '_'.join([cell_type, test.cmap2condition[drug], str(dose)])
                de_idx = np.where(test.var_names.isin(np.array(test.de_genes[pert_category])))[0]
                de_idx10 = np.where(test.var_names.isin(np.array(test.de_genes_10[pert_category])))[0]
                y_true = test.genes[list(idx),:]
                y_true_de = y_true[:,de_idx]
                y_true_de10 = y_true[:,de_idx10]

                sample_nums = y_true.shape[0] * coeff
                samples = []
                #print("coef:",coeff,"sample_num:",sample_nums)
                for name in sample_nums.index:
                    if name=='DMSO':
                        train_idx = train_drug_dict[name] & train_cell_types_dict[cell_type] & train_dosage_dict[1.0]
                    else:
                        train_idx = train_drug_dict[name] & train_cell_types_dict[cell_type] & train_dosage_dict[dose]
                    if len(train_idx)> 0:
                        samples += random.choices(train.genes[list(train_idx)],k=int(sample_nums[name]))
                        #print(len(samples))
                    else:
                        print(name, cell_type, drug)
                samples = torch.stack(samples)    
                samples_de = samples[:,de_idx]
                samples_de10 = samples[:,de_idx10]
                #print(type(samples))
                score = metr(y_true,samples)
                score_de = metr(y_true_de,samples_de)
                score_de10 = metr(y_true_de10,samples_de10)
                #print(score)
                drug_list.append(drug)
                cell_list.append(cell_type)
                dose_list.append(dose)
                evaluate_list.append(score)
                de.append(score_de)
                de10.append(score_de10)
    else:
        print(drug)   

result = pd.DataFrame({'drug':drug_list,'cell_type':cell_list,'dose':dose_list,
    'KNN':evaluate_list, 'de':de,'de10':de10})
result.to_csv('kNN.csv')
print(np.nanmean(result['KNN']), np.nanmean(result['de']), np.nanmean(result['de10']))       
        
        
    