# A pre-trained model for cellular responses to drug perturbations
## Introduction

Measuring single-cell RNA sequencing (scRNA-seq) responses to drug perturbations can facilitate drug development. However, an exhaust exploration of countless drugs is experimentally unfeasible, so computational tools are necessary to predict perturbations of unmeasured drugs. Existing tools are not designed for predicting scRNA-seq of unseen drugs. Here, we present a pre-trained model, which combines BERT, a wildly used pre-trained model, with autoencoder. The model learns transcriptional drug responses from bulk RNA sequencing in pre-training stage and can adapt to scRNA-seq in finetuning stage. It can predict scRNA-seq responses to drugs unseen in finetuning stage and model the heterogeneity of cellular responses.

## Code

The pretrain code is in pretrain.

The finetune code is in finetune_ae.

The baseline code is in newdrug_baseline.

